"""RNN architectures for approximating HMM forward filtering.

Each model takes a sequence of discrete emissions and produces a
predicted next-token probability distribution at every time step.

Model hierarchy
---------------
- **AbstractRNN** : Implements baseline functionality and abstract methods.
  - **ExactRNN**  : Implements the exact nonlinear forward-filter recurrence.
  - **ModelA**    : Linear recurrence with Cayley-stable A, stochastic readout.
    - **ModelAstar** : Model A with weights defined via a linearization. See ModelA.initialize_Astar.
  - **ModelB**    : Linear recurrence with Cayley-stable A, affine softmax readout.
"""

from abc import abstractmethod
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn import logsumexp

from .utils import DTYPE, params_to_stable_matrix, stable_matrix_to_params

# ---------------------------------------------------------------------------
# LossType enum
# ---------------------------------------------------------------------------


class LossType(str, Enum):
    """Training / evaluation objective."""
    EMISSIONS = "emissions"
    KL = "kl"
    HILBERT = "hilbert"


# ---------------------------------------------------------------------------
# Schema-driven parameter system
# ---------------------------------------------------------------------------


def _raw_shapes_for_entry(name, shape, constraint):
    """Return list of (raw_name, raw_shape) for a single schema entry."""
    if constraint == "stable":
        n = shape[0]
        return [
            (f"{name}_1", (n, n)),
            (f"{name}_2", (n * (n - 1) // 2,)),
        ]
    else:
        return [(name, shape)]


def _init_raw_weight(raw_name, raw_shape, constraint, key, ic_scale=0.01):
    """Initialize a single raw weight tensor."""
    if constraint == "stable":
        return jax.random.normal(key, raw_shape, dtype=DTYPE)
    else:
        return jax.random.normal(key, raw_shape, dtype=DTYPE) * ic_scale


def _transform_entry(name, constraint, raw_weights):
    """Apply the constraint transform for one schema entry."""
    if constraint == "unconstrained":
        return raw_weights[name]
    elif constraint == "stable":
        return params_to_stable_matrix(raw_weights[f"{name}_1"], raw_weights[f"{name}_2"], epsilon=1e-5)
    elif constraint == "stochastic":
        return jax.nn.softmax(raw_weights[name], axis=0)
    elif constraint == "nonneg":
        return raw_weights[name] ** 2
    else:
        raise ValueError(f"Unknown constraint: {constraint!r}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractRNN:
    """Base class for all RNN filtering models.

    Subclasses must implement:
      - ``schema(n, m)`` classmethod returning parameter specifications
      - ``integrate(...)`` static method defining the recurrence

    Construction: ``RNN(latent_dim=2, emission_dim=6, seed=0)``

    Weight access:
      - ``rnn.weights``      — transformed weights dict (e.g. ``{"A": ..., "B": ..., "C": ...}``)
      - ``rnn.raw_weights``  — raw backend weights dict (e.g. ``{"A_1": ..., "A_2": ..., "B": ..., "C": ...}``)
    """

    def __init__(self, latent_dim: int, emission_dim: int, seed: int = 0):
        self.latent_dim = latent_dim
        self.emission_dim = emission_dim
        self._schema = self.schema(latent_dim, emission_dim)

        key = jax.random.PRNGKey(seed)
        raw_entries = []
        for name, shape, constraint in self._schema:
            raw_entries.extend(_raw_shapes_for_entry(name, shape, constraint))

        keys = jax.random.split(key, len(raw_entries))
        self.raw_weights = {}
        for (raw_name, raw_shape), k in zip(raw_entries, keys):
            constraint = self._constraint_for_raw(raw_name)
            self.raw_weights[raw_name] = _init_raw_weight(raw_name, raw_shape, constraint, k)

        self.isFrozen = jax.tree.map(lambda _: False, self.raw_weights)

    def _constraint_for_raw(self, raw_name):
        """Look up the constraint type for a raw weight name."""
        for name, _shape, constraint in self._schema:
            if constraint == "stable":
                if raw_name in (f"{name}_1", f"{name}_2"):
                    return constraint
            else:
                if raw_name == name:
                    return constraint
        return "unconstrained"

    @classmethod
    @abstractmethod
    def schema(cls, n, m):
        """Return list of (name, shape, constraint) tuples defining weights."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def integrate(**kwargs):
        """Define the recurrence: ``(x_prev, emission_t) -> (x_t, y_t)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _warn_unknown_keys(self, keys, caller):
        unknown = set(keys) - set(self.raw_weights.keys())
        if unknown:
            import warnings
            warnings.warn(f"{caller}: unknown weight names {unknown}", stacklevel=3)

    def freeze(self, weight_names):
        """Prevent listed raw weights from being updated during training."""
        self._warn_unknown_keys(weight_names, "freeze")
        for key in self.isFrozen.keys() & weight_names:
            self.isFrozen[key] = True

    def unfreeze(self, weight_names):
        """Allow listed raw weights to be updated during training."""
        self._warn_unknown_keys(weight_names, "unfreeze")
        for key in self.isFrozen.keys() & weight_names:
            self.isFrozen[key] = False

    def set_raw_weights(self, raw_weights):
        """Overwrite raw weights by name."""
        self._warn_unknown_keys(raw_weights.keys(), "set_raw_weights")
        for key, value in raw_weights.items():
            if key in self.raw_weights:
                self.raw_weights[key] = jnp.asarray(value, dtype=DTYPE)

    @property
    def raw_weight_names(self):
        """Names of weights JAX is optimizing over. Depending on constraints, this can be a larger list than weight_names."""
        return list(self.raw_weights.keys())

    @property
    def weight_names(self):
        """Names of weights defined in the schema."""
        return [name for name, _shape, _constraint in self._schema]

    @property
    def weights(self):
        """Constrained weights dict (schema names -> constrained arrays)."""
        return self._get_constrained_weights(self.raw_weights)

    # ------------------------------------------------------------------
    # Schema-driven transforms
    # ------------------------------------------------------------------

    @classmethod
    def _get_constrained_weights(cls, raw_weights):
        """Apply constraint transforms to raw weights. Implemented once in base."""

        # Uses cls.schema(0, 0) to get (name, constraint) pairs, since this
        # is a cls method and has no reference to self._schema. Shapes are
        # irrelevant for the transform logic and are inferred from the weights.
        schema = cls.schema(0, 0)
        result = {}
        for name, _shape, constraint in schema:
            result[name] = _transform_entry(name, constraint, raw_weights)
        return result

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    def _resolve_loss(self, loss):
        """Return (loss_fn, needs_posterior) for a LossType enum value."""
        loss = LossType(loss)
        cls = type(self)
        if loss is LossType.EMISSIONS:
            return cls._expected_surprisal, False
        elif loss is LossType.KL:
            return cls._expected_kl_divergence, True
        elif loss is LossType.HILBERT:
            return cls._expected_hilbert_distance, True
        raise ValueError(f"Unknown loss: {loss!r}")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _forward_scan(cls, raw_weights, emissions, x0):
        """Run the RNN on a single emission sequence."""
        w = cls._get_constrained_weights(raw_weights)

        def step(x_prev, emission_t):
            x_t, y_t = cls.integrate(x_prev=x_prev, emission_t=emission_t, **w)
            return x_t, (y_t, x_t)

        _, (Y, X) = jax.lax.scan(step, x0, emissions)
        return Y, X

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _batched_forward(cls, raw_weights, emissions, x0):
        """Vectorised forward pass over a batch of sequences."""
        return jax.vmap(cls._forward_scan, in_axes=(None, 0, None))(raw_weights, emissions, x0)

    def predict(self, emissions, x0=None):
        """Predict next-token distributions for a batch of emission sequences.

        Parameters
        ----------
        emissions : array-like, shape (B, T)
        x0 : array-like, shape (latent_dim,), optional
            Initial hidden state. Defaults to zeros.

        Returns
        -------
        Y : array, shape (B, T, emission_dim)
        X : array, shape (B, T, latent_dim)
        """
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        x0 = self._resolve_x0(x0)
        output, latent = self._batched_forward(self.raw_weights, emissions, x0)
        return jnp.array(output), jnp.array(latent)

    def loss(self, emissions, loss=LossType.EMISSIONS, x0=None):
        """Compute a loss on a batch of emissions.

        Parameters
        ----------
        emissions : array-like, shape (B, T)
        loss : LossType
            One of ``LossType.EMISSIONS``, ``LossType.KL``, ``LossType.HILBERT``.
            For ``KL`` and ``HILBERT``, ``emissions`` must be accompanied
            by ``targets`` — use :meth:`eval_loss` for the convenience wrapper.
        x0 : array-like, shape (latent_dim,), optional
        """
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        x0 = self._resolve_x0(x0)
        loss_fn, _ = self._resolve_loss(loss)
        return loss_fn(self.raw_weights, emissions, x0)

    # ------------------------------------------------------------------
    # LossType functions (private, jitted)
    # ------------------------------------------------------------------

    @classmethod
    @partial(jax.jit, static_argnums=(0, 4))
    def _expected_surprisal(cls, raw_weights, emissions, x0, do_average=True):
        """Mean negative log-likelihood of the next token."""
        output, _ = cls._batched_forward(raw_weights, emissions, x0)
        output = output[:, :-1, :]
        next_token = emissions[:, 1:]
        probs = jnp.take_along_axis(output, next_token[..., None], axis=-1).squeeze(-1)
        nll = -jnp.log(probs + 1e-32)
        return jnp.mean(nll) if do_average else nll

    @classmethod
    @partial(jax.jit, static_argnums=(0, 5))
    def _expected_kl_divergence(cls, raw_weights, emissions, targets, x0, do_average=True):
        """KL divergence from the ground-truth posterior to the RNN prediction."""
        p_rnn, _ = cls._batched_forward(raw_weights, emissions, x0)
        eps = 1e-12
        p_rnn = p_rnn / jnp.sum(p_rnn, axis=-1, keepdims=True)
        p_target = targets / jnp.sum(targets, axis=-1, keepdims=True)
        kl = jnp.sum(p_target * (jnp.log(p_target + eps) - jnp.log(p_rnn + eps)), axis=-1)
        return jnp.mean(kl) if do_average else kl

    @classmethod
    @partial(jax.jit, static_argnums=(0, 5))
    def _expected_hilbert_distance(cls, raw_weights, emissions, targets, x0, do_average=True):
        """Hilbert projective metric between RNN output and target distributions."""
        p_rnn, _ = cls._batched_forward(raw_weights, emissions, x0)
        eps = 1e-16
        p_rnn = jnp.clip(p_rnn, eps, None)
        p_target = jnp.clip(targets, eps, None)
        log_ratio = jnp.log(p_rnn) - jnp.log(p_target)
        hilbert = jnp.max(log_ratio, axis=-1) - jnp.min(log_ratio, axis=-1)
        return jnp.mean(hilbert) if do_average else hilbert

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, hmm, *, loss=LossType.EMISSIONS, batch_size=100, time_steps=1000,
        num_epochs=1, learning_rate=2e-2, optimization_steps=2000,
        print_every=100, x0=None,
    ):
        """Train the model.

        Parameters
        ----------
        hmm : DiscreteHMM
        loss : LossType
            ``LossType.EMISSIONS`` — minimise next-token negative log-likelihood.
            ``LossType.KL``        — minimise KL divergence to exact HMM posterior.
            ``LossType.HILBERT``   — minimise Hilbert projective metric to posterior.
        """
        loss_fn, needs_posterior = self._resolve_loss(loss)
        return self._train(
            hmm,
            loss_fn=loss_fn,
            needs_posterior=needs_posterior,
            batch_size=batch_size,
            time_steps=time_steps,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimization_steps=optimization_steps,
            print_every=print_every,
            x0=x0,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_loss(self, hmm, *, loss=LossType.EMISSIONS, batch_size=100, time_steps=1000, x0=None):
        """Per-timestep loss (not averaged) on freshly sampled data.

        Parameters
        ----------
        hmm : DiscreteHMM
        loss : LossType
            ``LossType.EMISSIONS``, ``LossType.KL``, or ``LossType.HILBERT``.
        """
        loss_fn, needs_posterior = self._resolve_loss(loss)
        _, emissions = hmm.sample(batch_size, time_steps)
        emissions_jax = jnp.asarray(emissions, dtype=jnp.int32)
        x0_vec = self._resolve_x0(x0)
        if needs_posterior:
            _, targets = hmm.compute_posterior(emissions)
            return loss_fn(self.raw_weights, emissions_jax, targets, x0_vec, do_average=False)
        return loss_fn(self.raw_weights, emissions_jax, x0_vec, do_average=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_x0(self, x0):
        if x0 is None:
            return jnp.zeros(self.latent_dim, dtype=DTYPE)
        return jnp.asarray(x0, dtype=DTYPE)

    def _train(
        self, hmm, *, loss_fn, needs_posterior, batch_size, time_steps,
        num_epochs, learning_rate, optimization_steps, print_every, x0=None,
    ):
        loss_history = np.zeros((optimization_steps, num_epochs))
        x0_vec = self._resolve_x0(x0)

        for epoch in range(num_epochs):
            _, emissions = hmm.sample(batch_size, time_steps)
            emissions_jax = jnp.asarray(emissions, dtype=jnp.int32)

            optimizer = optax.chain(optax.adam(learning_rate), optax.masked(optax.set_to_zero(), self.isFrozen))
            opt_state = optimizer.init(self.raw_weights)

            if needs_posterior:
                _, targets = hmm.compute_posterior(emissions)
                def update(raw_weights, state):
                    val, grads = jax.value_and_grad(loss_fn)(raw_weights, emissions_jax, targets, x0_vec)
                    updates, state = optimizer.update(grads, state, raw_weights)
                    raw_weights = optax.apply_updates(raw_weights, updates)
                    return raw_weights, state, val
            else:
                def update(raw_weights, state):
                    val, grads = jax.value_and_grad(loss_fn)(raw_weights, emissions_jax, x0_vec)
                    updates, state = optimizer.update(grads, state, raw_weights)
                    raw_weights = optax.apply_updates(raw_weights, updates)
                    return raw_weights, state, val

            for s in range(1, optimization_steps + 1):
                self.raw_weights, opt_state, current_loss = update(self.raw_weights, opt_state)
                loss_history[s - 1, epoch] = current_loss
                if s % print_every == 0:
                    print(f"epoch {epoch}, step {s}: loss={float(current_loss):.12e}")

        return loss_history


# ---------------------------------------------------------------------------
# ExactRNN  (ground-truth nonlinear filter)
# ---------------------------------------------------------------------------


class ExactRNN(AbstractRNN):
    """Exact forward-filter dynamics: ``x_t = log(A exp(x_{t-1})) + B[:, y_t]``.

    Raw weights A and C are stored in unconstrained log-space and
    softmax-normalised to column-stochastic matrices before the scan.
    """

    @classmethod
    def schema(cls, n, m):
        return [
            ("A", (n, n), "stochastic"),
            ("B", (n, m), "unconstrained"),
            ("C", (m, n), "stochastic"),
        ]

    @staticmethod
    def integrate(A, B, C, x_prev, emission_t):
        x_t = jnp.log(A @ jnp.exp(x_prev)) + B[:, emission_t]
        x_t = x_t - logsumexp(x_t)
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_weights(self, hmm):
        """Set raw weights to the true HMM values."""
        self.set_raw_weights({
            "A": np.log(hmm.transfer_matrix),
            "B": np.log(hmm.emission_matrix.T),
            "C": np.log(hmm.emission_matrix @ hmm.transfer_matrix),
        })


# ---------------------------------------------------------------------------
# ModelA  (stable linear recurrence, stochastic readout)
# ---------------------------------------------------------------------------


class ModelA(AbstractRNN):
    """Linear RNN: ``x_t = A x_{t-1} + B[:, y_t]``; ``p_t = C softmax(x_t)``.

    A is parameterised via the Cayley transform to guarantee spectral
    radius <= 1.  C is softmax-normalised to a column-stochastic matrix.
    """

    @classmethod
    def schema(cls, n, m):
        return [
            ("A", (n, n), "stable"),
            ("B", (n, m), "unconstrained"),
            ("C", (m, n), "stochastic"),
        ]

    @staticmethod
    def integrate(A, B, C, x_prev, emission_t):
        x_t = A @ x_prev + B[:, emission_t]
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_Astar(self, hmm):
        """Initialize raw weights from an HMM via log-probability linearization.

        Computes the Jacobian of the log-transfer map at the stationary
        distribution and sets A, B, C accordingly.
        """
        T = hmm.transfer_matrix
        E = hmm.emission_matrix
        p = hmm.latent_stationary_density

        Tp = T @ p
        J = (T * p[None, :]) / Tp[:, None]
        f = np.log(T @ p)
        bias = f - J @ np.log(p)

        A_mat = J
        B_mat = np.log(E.T) + bias[:, None]
        C_mat = E @ T

        A1, A2 = stable_matrix_to_params(jnp.asarray(A_mat, dtype=DTYPE))
        self.set_raw_weights({
            "A_1": np.array(A1),
            "A_2": np.array(A2),
            "B": B_mat,
            "C": np.log(C_mat),
        })


# ---------------------------------------------------------------------------
# ModelB  (stable linear recurrence, affine softmax readout)
# ---------------------------------------------------------------------------


class ModelB(AbstractRNN):
    """Linear RNN with bias in readout: ``x_t = A x_{t-1} + B[:, y_t]``; ``p_t = softmax(C x_t + d)``.

    A is Cayley-stable.  The output bias ``d`` allows a richer readout mapping.
    """

    @classmethod
    def schema(cls, n, m):
        return [
            ("A", (n, n), "stable"),
            ("B", (n, m), "unconstrained"),
            ("C", (m, n), "unconstrained"),
            ("d", (m,), "unconstrained"),
        ]

    @staticmethod
    def integrate(A, B, C, d, x_prev, emission_t):
        x_t = A @ x_prev + B[:, emission_t]
        y_t = jax.nn.softmax(C @ x_t + d)
        return x_t, y_t
