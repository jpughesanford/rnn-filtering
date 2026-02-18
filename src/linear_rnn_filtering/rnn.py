"""RNN architectures for approximating HMM forward filtering.

Each model takes a sequence of discrete emissions and produces a
predicted next-token probability distribution at every time step.

"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn import logsumexp
from jax.typing import ArrayLike

from .hmm import DiscreteHMM
from .types import ConstraintType, LossType
from .utils import DTYPE, params_to_stable_matrix, stable_matrix_to_params

__all__ = ["AbstractRNN", "ExactRNN", "ModelA", "ModelB"]


def _validate_scheme(schema: list[tuple[str, tuple[int,...], str]]) -> bool:
    """Check whether a weight schema is valid.

    A valid schema is a list of ``(name, shape, constraint)`` tuples where
    names are unique strings, shapes are tuples of positive integers, and
    constraints are :class:`ConstraintType` values. ``STABLE`` constraints
    additionally require the shape to be square.

    Args:
        schema (list[tuple[str, tuple[int], ConstraintType]]): Weight schema to validate.

    Returns:
        bool: True if the schema is valid, False otherwise.
    """
    names = set()
    for entry in schema:
        if len(entry) != 3:
            return False
        name, shape, constraint = entry
        if not isinstance(name, str):
            return False
        if name in names:
            return False
        names.add(name)
        if not isinstance(shape, tuple):
            return False
        if any(d <= 0 for d in shape):
            return False
        if not isinstance(constraint, ConstraintType):
            return False
        if constraint is ConstraintType.STABLE and (len(shape) != 2 or shape[0] != shape[1]):
            return False
    return True


def _weights_needed_to_enforce_constraint(
    name: str, shape: tuple[int], constraint: ConstraintType = ConstraintType.UNCONSTRAINED
) -> list[tuple[str, tuple[int]]]:
    """Return the raw parameter names and shapes needed for a single schema entry.

    For most constraints the raw parameter has the same name and shape.
    The ``STABLE`` constraint requires two raw parameters (one square matrix
    and one vector of lower-triangular entries) to parameterise a
    Cayley-stable matrix.

    Args:
        name (str): Parameter name from the schema.
        shape (tuple[int]): Shape of the constrained parameter.
        constraint (ConstraintType, optional): Constraint type. Defaults to ``ConstraintType.UNCONSTRAINED``.

    Returns:
        list[tuple[str, tuple[int]]]: List of ``(raw_name, raw_shape)`` pairs.
    """
    match constraint:
        case ConstraintType.STABLE:
            n = shape[0]
            return [
                (f"{name}_1", shape),
                (f"{name}_2", (n * (n - 1) // 2,)),
            ]
        case _:
            return [(name, shape)]


def _compute_constrained_weight(name: str, constraint: ConstraintType, raw_weights: dict[str, jax.Array]) -> jax.Array:
    """Apply the constraint transform for one schema entry.

    Maps raw (unconstrained) weight arrays to a constrained weight array
    according to the specified constraint type.

    Args:
        name (str): Parameter name from the schema.
        constraint (ConstraintType): Constraint type to apply.
        raw_weights (dict[str, jax.Array]): Dictionary of raw weight arrays.

    Returns:
        jax.Array: Constrained weight array.
    """
    match constraint:
        case ConstraintType.UNCONSTRAINED:
            return raw_weights[name]
        case ConstraintType.STABLE:
            return params_to_stable_matrix(raw_weights[f"{name}_1"], raw_weights[f"{name}_2"], epsilon=1e-5)
        case ConstraintType.STOCHASTIC:
            return jax.nn.softmax(raw_weights[name], axis=0)
        case ConstraintType.NONNEGATIVE:
            return raw_weights[name] ** 2


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractRNN(metaclass=ABCMeta):
    """Base class for all RNN filtering models.

    Subclasses must implement:
        - ``schema(latent_dim, emission_dim)`` (staticmethod) returning parameter specifications.
        - ``integrate(..., x_prev, emission_t)`` (staticmethod) defining the recurrence.

    Attributes:
        latent_dim (int): Dimensionality of the latent state.
        emission_dim (int): Number of discrete emissions.
        raw_weights (dict[str, jax.Array]): Raw (unconstrained) weight arrays keyed by name.
        isFrozen (dict[str, bool]): Boolean map indicating whether each raw weight is frozen during training.
    """

    def __init__(self, latent_dim: int, emission_dim: int, seed: int = 0, ic_scale: float = 0.01):
        """Initialize RNN weights according to schema and (optional) PRNG seed.

        Args:
            latent_dim (int): Dimensionality of the RNN latent state.
            emission_dim (int): Dimensionality of the RNN output.
            seed (int, optional): Random seed for weight initialization. Defaults to 0.
            ic_scale (float, optional): Standard deviation of the initial weight distribution. Defaults to 0.01.

        Raises:
            ValueError: If the schema returned by the subclass is invalid.
        """
        self.latent_dim: int = latent_dim
        self.emission_dim: int = emission_dim

        self._schema: list[tuple[str, tuple[int], ConstraintType]] = self.schema(latent_dim, emission_dim)
        if not _validate_scheme(self._schema):
            raise ValueError("Scheme is invalid.")

        self.seed: int = seed

        # compute number of raw weights needed and
        # split rng key into parts for each raw weight
        prng_key = jax.random.PRNGKey(seed)
        raw_schema = [
            raw_entry for entry in self._schema for raw_entry in _weights_needed_to_enforce_constraint(*entry)
        ]
        keys = jax.random.split(prng_key, len(raw_schema))

        # Initialize
        self.raw_weights: dict[str, jax.Array] = {
            raw_name: jax.random.normal(key, raw_shape, dtype=DTYPE) * ic_scale
            for (raw_name, raw_shape), key in zip(raw_schema, keys, strict=True)
        }
        self.isFrozen: dict[str, bool] = dict.fromkeys(self.raw_weights, False)

    @staticmethod
    @abstractmethod
    def schema(latent_dim: int, emission_dim: int) -> list[tuple[str, tuple[int,...], ConstraintType]]:
        """Return the weight schema for this architecture.

        Each entry is a ``(name, shape, constraint)`` tuple specifying one
        logical network weight.

        Args:
            latent_dim (int): Latent state dimensionality.
            emission_dim (int): Emission dimensionality.

        Returns:
            list[tuple[str, tuple[int,...], ConstraintType]]: Weight schema.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def integrate(**kwargs):
        """Define the recurrence: ``(x_prev, emission_t) -> (x_t, y_t)``.

        Subclasses should define the signature as
        ``integrate(W1, W2, ..., x_prev, emission_t)`` where the weight
        arguments match the schema entries in order, followed by the
        previous latent state and the current emission.

        Args:
            x_prev (jax.Array): Previous hidden state of shape (latent_dim,).
            emission_t (jax.Array): Current emission index (scalar int32).
            **kwargs: Constrained weight arrays as defined by :meth:`schema`.

        Returns:
            x_t (jax.Array): Updated hidden state of shape (latent_dim,).
            y_t (jax.Array): Predicted next-emission distribution of shape (emission_dim,).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _warn_unknown_keys(self, keys: Iterable[str], caller: str) -> None:
        """Emit a warning if any keys are not recognized raw weight names.

        Args:
            keys (Iterable[str]): Weight names to check.
            caller (str): Name of the calling method, used in the warning message.
        """
        unknown = set(keys) - set(self.raw_weights.keys())
        if unknown:
            import warnings

            warnings.warn(f"{caller}: unknown weight names {unknown}", stacklevel=3)

    def freeze(self, weight_names: Iterable[str]) -> None:
        """Prevent listed raw weights from being updated during training.

        Frozen weights have their gradients zeroed out by the optimizer.

        Args:
            weight_names (Iterable[str]): Names of raw weights to freeze.
        """
        self._warn_unknown_keys(weight_names, "freeze")
        for key in self.isFrozen.keys() & weight_names:
            self.isFrozen[key] = True

    def unfreeze(self, weight_names: Iterable[str]) -> None:
        """Allow listed raw weights to be updated during training.

        Args:
            weight_names (Iterable[str]): Names of raw weights to unfreeze.
        """
        self._warn_unknown_keys(weight_names, "unfreeze")
        for key in self.isFrozen.keys() & weight_names:
            self.isFrozen[key] = False

    def set_raw_weights(self, raw_weights: dict[str, ArrayLike]) -> None:
        """Overwrite raw weights by name.

        Values are cast to the global ``DTYPE``. Unknown names trigger a warning
        and are silently skipped.

        Args:
            raw_weights (dict[str, ArrayLike]): Mapping from raw weight names to new values.
        """
        self._warn_unknown_keys(raw_weights.keys(), "set_raw_weights")
        for key, value in raw_weights.items():
            if key in self.raw_weights:
                self.raw_weights[key] = jnp.asarray(value, dtype=DTYPE)

    @property
    def raw_weight_names(self) -> list[str]:
        """Names of raw weights JAX is optimizing over.

        Depending on constraints, this can be a larger list than
        :attr:`weight_names` (e.g. a ``STABLE`` matrix produces two raw entries).

        Returns:
            list[str]: Raw weight names.
        """
        return list(self.raw_weights.keys())

    @property
    def weight_names(self) -> list[str]:
        """Names of logical weights defined in the schema.

        Returns:
            list[str]: Schema weight names.
        """
        return [name for name, _shape, _constraint in self._schema]

    @property
    def weights(self) -> dict[str, jax.Array]:
        """Constrained weights dict mapping schema names to constrained arrays.

        Returns:
            dict[str, jax.Array]: Constrained weight arrays.
        """
        return self._get_constrained_weights(self.raw_weights)

    # ------------------------------------------------------------------
    # Schema-driven transforms
    # ------------------------------------------------------------------

    @classmethod
    def _get_constrained_weights(cls, raw_weights: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Apply constraint transforms to all raw weights.

        Uses ``cls.schema(0, 0)`` to obtain ``(name, constraint)`` pairs.
        Shapes are inferred from the raw weight arrays themselves.

        Args:
            raw_weights (dict[str, jax.Array]): Raw (unconstrained) weight arrays.

        Returns:
            dict[str, jax.Array]: Constrained weight arrays keyed by schema name.
        """
        schema = cls.schema(0, 0)
        result = {}
        for name, _shape, constraint in schema:
            result[name] = _compute_constrained_weight(name, constraint, raw_weights)
        return result

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    def _resolve_loss(self, loss: LossType) -> tuple[Callable, bool]:
        """Look up the JIT-compiled loss function for a given :class:`LossType`.

        Args:
            loss (LossType): Loss type enum value.

        Returns:
            loss_fn (callable): JIT-compiled loss function.
            needs_posterior (bool): Whether the loss requires ground-truth posterior targets.

        Raises:
            ValueError: If ``loss`` is not a recognized :class:`LossType`.
        """
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
    def _forward_scan(
        cls, raw_weights: dict[str, jax.Array], emissions: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Run the RNN on a single emission sequence via ``jax.lax.scan``.

        Args:
            raw_weights (dict[str, jax.Array]): Raw weight arrays.
            emissions (jax.Array): Emission indices of shape (T,), dtype int32.
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            Y (jax.Array): Predicted distributions of shape (T, emission_dim).
            X (jax.Array): Hidden states of shape (T, latent_dim).
        """
        w = cls._get_constrained_weights(raw_weights)

        def step(x_prev, emission_t):
            x_t, y_t = cls.integrate(x_prev=x_prev, emission_t=emission_t, **w)
            return x_t, (y_t, x_t)

        _, (Y, X) = jax.lax.scan(step, x0, emissions)
        return Y, X

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _batched_forward(
        cls, raw_weights: dict[str, jax.Array], emissions: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Vectorised forward pass over a batch of sequences via ``jax.vmap``.

        Args:
            raw_weights (dict[str, jax.Array]): Raw weight arrays.
            emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            Y (jax.Array): Predicted distributions of shape (B, T, emission_dim).
            X (jax.Array): Hidden states of shape (B, T, latent_dim).
        """
        return jax.vmap(cls._forward_scan, in_axes=(None, 0, None))(raw_weights, emissions, x0)

    def predict(self, emissions: ArrayLike, x0: ArrayLike | None = None) -> tuple[jax.Array, jax.Array]:
        """Predict next-token distributions for a batch of emission sequences.

        Args:
            emissions (ArrayLike): Observed emission indices of shape (B, T).
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            Y (jax.Array): Predicted next-emission distributions of shape (B, T, emission_dim).
            X (jax.Array): Hidden state trajectories of shape (B, T, latent_dim).
        """
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        x0 = self._resolve_x0(x0)
        output, latent = self._batched_forward(self.raw_weights, emissions, x0)
        return jnp.array(output), jnp.array(latent)

    def loss(self, emissions: ArrayLike, loss: LossType = LossType.EMISSIONS, x0: ArrayLike | None = None) -> jax.Array:
        """Compute a scalar loss on a batch of emissions.

        Args:
            emissions (ArrayLike): Observed emission indices of shape (B, T).
            loss (LossType, optional): Loss type. One of ``LossType.EMISSIONS``,
                ``LossType.KL``, or ``LossType.HILBERT``. For ``KL`` and ``HILBERT``,
                ``emissions`` must be accompanied by ``targets`` — use :meth:`eval_loss`
                for the convenience wrapper. Defaults to ``LossType.EMISSIONS``.
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            jax.Array: Scalar loss value.
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
    def _expected_surprisal(
        cls, raw_weights: dict[str, jax.Array], emissions: jax.Array, x0: jax.Array, do_average: bool = True
    ) -> jax.Array:
        """Mean negative log-likelihood of the next token.

        Computes ``-log p(y_{t+1} | y_{0:t})`` averaged over batch and time.

        Args:
            raw_weights (dict[str, jax.Array]): Raw weight arrays.
            emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).
            do_average (bool, optional): If True, return the scalar mean. If False,
                return per-timestep values. Defaults to True.

        Returns:
            jax.Array: Scalar mean NLL if ``do_average`` is True, otherwise array of shape (B, T-1).
        """
        output, _ = cls._batched_forward(raw_weights, emissions, x0)
        output = output[:, :-1, :]
        next_token = emissions[:, 1:]
        probs = jnp.take_along_axis(output, next_token[..., None], axis=-1).squeeze(-1)
        nll = -jnp.log(probs + 1e-32)
        return jnp.mean(nll) if do_average else nll

    @classmethod
    @partial(jax.jit, static_argnums=(0, 5))
    def _expected_kl_divergence(
        cls,
        raw_weights: dict[str, jax.Array],
        emissions: jax.Array,
        targets: jax.Array,
        x0: jax.Array,
        do_average: bool = True,
    ) -> jax.Array:
        """KL divergence from the ground-truth posterior to the RNN prediction.

        Computes ``KL(p_target || p_rnn)`` at each time step.

        Args:
            raw_weights (dict[str, jax.Array]): Raw weight arrays.
            emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
            targets (jax.Array): Ground-truth posterior distributions of shape (B, T, emission_dim).
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).
            do_average (bool, optional): If True, return the scalar mean. If False,
                return per-timestep values. Defaults to True.

        Returns:
            jax.Array: Scalar mean KL if ``do_average`` is True, otherwise array of shape (B, T).
        """
        p_rnn, _ = cls._batched_forward(raw_weights, emissions, x0)
        eps = 1e-12
        p_rnn = p_rnn / jnp.sum(p_rnn, axis=-1, keepdims=True)
        p_target = targets / jnp.sum(targets, axis=-1, keepdims=True)
        kl = jnp.sum(p_target * (jnp.log(p_target + eps) - jnp.log(p_rnn + eps)), axis=-1)
        return jnp.mean(kl) if do_average else kl

    @classmethod
    @partial(jax.jit, static_argnums=(0, 5))
    def _expected_hilbert_distance(
        cls,
        raw_weights: dict[str, jax.Array],
        emissions: jax.Array,
        targets: jax.Array,
        x0: jax.Array,
        do_average: bool = True,
    ) -> jax.Array:
        """Hilbert projective metric between RNN output and target distributions.

        The Hilbert metric is defined as
        ``max_i log(p_rnn_i / p_target_i) - min_i log(p_rnn_i / p_target_i)``.

        Args:
            raw_weights (dict[str, jax.Array]): Raw weight arrays.
            emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
            targets (jax.Array): Ground-truth posterior distributions of shape (B, T, emission_dim).
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).
            do_average (bool, optional): If True, return the scalar mean. If False,
                return per-timestep values. Defaults to True.

        Returns:
            jax.Array: Scalar mean Hilbert distance if ``do_average`` is True, otherwise array of shape (B, T).
        """
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
        self,
        hmm: DiscreteHMM,
        *,
        loss: LossType = LossType.EMISSIONS,
        batch_size: int = 100,
        time_steps: int = 1000,
        num_epochs: int = 1,
        learning_rate: float = 2e-2,
        optimization_steps: int = 2000,
        print_every: int = 100,
        x0: ArrayLike | None = None,
    ) -> np.ndarray:
        """Train the model on data sampled from an HMM.

        At each epoch, a fresh batch of trajectories is sampled from ``hmm``.
        An Adam optimizer is used with optional weight freezing.

        Args:
            hmm (DiscreteHMM): HMM instance used to generate training data.
            loss (LossType, optional): Training objective. One of ``LossType.EMISSIONS``,
                ``LossType.KL``, or ``LossType.HILBERT``. Defaults to ``LossType.EMISSIONS``.
            batch_size (int, optional): Number of independent trajectories per epoch. Defaults to 100.
            time_steps (int, optional): Length of each sampled trajectory. Defaults to 1000.
            num_epochs (int, optional): Number of data-resampling epochs. Defaults to 1.
            learning_rate (float, optional): Adam learning rate. Defaults to 2e-2.
            optimization_steps (int, optional): Gradient steps per epoch. Defaults to 2000.
            print_every (int, optional): Print loss every this many steps. Defaults to 100.
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            loss_history (numpy.ndarray): Loss values of shape (optimization_steps, num_epochs).
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

    def eval_loss(
        self,
        hmm: DiscreteHMM,
        *,
        loss: LossType = LossType.EMISSIONS,
        batch_size: int = 100,
        time_steps: int = 1000,
        x0: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate per-timestep loss (not averaged) on freshly sampled data.

        Samples a new batch of trajectories from ``hmm`` and returns the
        unaggregated loss at every time step, which is useful for
        analysing convergence along the sequence.

        Args:
            hmm (DiscreteHMM): HMM instance used to generate evaluation data.
            loss (LossType, optional): Evaluation objective. One of ``LossType.EMISSIONS``,
                ``LossType.KL``, or ``LossType.HILBERT``. Defaults to ``LossType.EMISSIONS``.
            batch_size (int, optional): Number of independent trajectories. Defaults to 100.
            time_steps (int, optional): Length of each sampled trajectory. Defaults to 1000.
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            jax.Array: Per-timestep loss values of shape (B, T) or (B, T-1) depending on loss type.
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

    def _resolve_x0(self, x0: ArrayLike | None) -> jax.Array:
        """Return a JAX array for the initial hidden state, defaulting to zeros.

        Args:
            x0 (ArrayLike or None): Initial hidden state. If None, returns a zero vector.

        Returns:
            jax.Array: Initial hidden state of shape (latent_dim,).
        """
        if x0 is None:
            return jnp.zeros(self.latent_dim, dtype=DTYPE)
        return jnp.asarray(x0, dtype=DTYPE)

    def _train(
        self,
        hmm: DiscreteHMM,
        *,
        loss_fn: Callable,
        needs_posterior: bool,
        batch_size: int,
        time_steps: int,
        num_epochs: int,
        learning_rate: float,
        optimization_steps: int,
        print_every: int,
        x0: ArrayLike | None = None,
    ) -> np.ndarray:
        """Inner training loop called by :meth:`train`.

        Handles data sampling, optimizer construction, and gradient updates.
        Frozen weights are masked via ``optax.set_to_zero()``.

        Args:
            hmm (DiscreteHMM): HMM instance used to generate training data.
            loss_fn (Callable): JIT-compiled loss function.
            needs_posterior (bool): Whether the loss requires ground-truth posterior targets.
            batch_size (int): Number of independent trajectories per epoch.
            time_steps (int): Length of each sampled trajectory.
            num_epochs (int): Number of data-resampling epochs.
            learning_rate (float): Adam learning rate.
            optimization_steps (int): Gradient steps per epoch.
            print_every (int): Print loss every this many steps.
            x0 (array-like, optional): Initial hidden state. Defaults to zeros.

        Returns:
            loss_history (numpy.ndarray): Loss values of shape (optimization_steps, num_epochs).
        """
        loss_history = np.zeros((optimization_steps, num_epochs))
        x0_vec = self._resolve_x0(x0)

        for epoch in range(num_epochs):
            _, emissions = hmm.sample(batch_size, time_steps)
            emissions_jax = jnp.asarray(emissions, dtype=jnp.int32)

            optimizer = optax.chain(optax.adam(learning_rate), optax.masked(optax.set_to_zero(), self.isFrozen))
            opt_state = optimizer.init(self.raw_weights)

            if needs_posterior:
                _, targets = hmm.compute_posterior(emissions)

                def update(
                    raw_weights, state,
                    emissions_jax=emissions_jax, targets=targets, optimizer=optimizer,
                ):
                    val, grads = jax.value_and_grad(loss_fn)(raw_weights, emissions_jax, targets, x0_vec)
                    updates, state = optimizer.update(grads, state, raw_weights)
                    raw_weights = optax.apply_updates(raw_weights, updates)
                    return raw_weights, state, val
            else:

                def update(
                    raw_weights, state,
                    emissions_jax=emissions_jax, optimizer=optimizer,
                ):
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

    Raw weights ``A`` and ``C`` are stored in unconstrained log-space and
    softmax-normalised to column-stochastic matrices before the scan.
    """

    @staticmethod
    def schema( n: int, m: int) -> list[tuple[str, tuple[int,...], ConstraintType]]:
        """Return the weight schema for the exact filter.

        Args:
            n (int): Latent state dimensionality.
            m (int): Emission dimensionality.

        Returns:
            list[tuple[str, tuple[int], ConstraintType]]: Weight schema with entries
                for ``A`` (stochastic), ``B`` (unconstrained), and ``C`` (stochastic).
        """
        return [
            ("A", (n, n), ConstraintType.STOCHASTIC),
            ("B", (n, m), ConstraintType.UNCONSTRAINED),
            ("C", (m, n), ConstraintType.STOCHASTIC),
        ]

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Exact nonlinear forward-filter recurrence.

        Computes ``x_t = log(A exp(x_{t-1})) + B[:, y_t]`` (normalised in
        log-space) and reads out via ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Column-stochastic transfer matrix of shape (n, n).
            B (jax.Array): Log-emission matrix of shape (n, m).
            C (jax.Array): Column-stochastic readout matrix of shape (m, n).
            x_prev (jax.Array): Previous log-posterior of shape (n,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated log-posterior of shape (n,).
            y_t (jax.Array): Predicted next-emission distribution of shape (m,).
        """
        x_t = jnp.log(A @ jnp.exp(x_prev)) + B[:, emission_t]
        x_t = x_t - logsumexp(x_t)
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_weights(self, hmm: DiscreteHMM) -> None:
        """Set raw weights to the true HMM parameter values.

        This produces an RNN that exactly reproduces the HMM forward
        filter, and is useful as an upper-bound baseline.

        Args:
            hmm (DiscreteHMM): Source HMM whose parameters are copied.
        """
        self.set_raw_weights(
            {
                "A": np.log(hmm.transfer_matrix),
                "B": np.log(hmm.emission_matrix.T),
                "C": np.log(hmm.emission_matrix @ hmm.transfer_matrix),
            }
        )


# ---------------------------------------------------------------------------
# ModelA  (stable linear recurrence, stochastic readout)
# ---------------------------------------------------------------------------


class ModelA(AbstractRNN):
    """Linear RNN with stable latent dynamics and stochastic readout.

    Recurrence: ``x_t = A x_{t-1} + B[:, y_t]``;
    readout: ``p_t = C softmax(x_t)``.

    ``A`` is parameterised via the Cayley transform to guarantee spectral
    radius <= 1.  ``C`` is softmax-normalised to a column-stochastic matrix.
    """

    @staticmethod
    def schema(n: int, m: int) -> list[tuple[str, tuple[int,...], ConstraintType]]:
        """Return the weight schema for Model A.

        Args:
            n (int): Latent state dimensionality.
            m (int): Emission dimensionality.

        Returns:
            list[tuple[str, tuple[int], ConstraintType]]: Weight schema with entries
                for ``A`` (stable), ``B`` (unconstrained), and ``C`` (stochastic).
        """
        return [
            ("A", (n, n), ConstraintType.STABLE),
            ("B", (n, m), ConstraintType.UNCONSTRAINED),
            ("C", (m, n), ConstraintType.STOCHASTIC),
        ]

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with stochastic readout.

        Computes ``x_t = A x_{t-1} + B[:, y_t]`` and reads out via
        ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape (n, n).
            B (jax.Array): Input embedding matrix of shape (n, m).
            C (jax.Array): Column-stochastic readout matrix of shape (m, n).
            x_prev (jax.Array): Previous hidden state of shape (n,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated hidden state of shape (n,).
            y_t (jax.Array): Predicted next-emission distribution of shape (m,).
        """
        x_t = A @ x_prev + B[:, emission_t]
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_astar(self, hmm: DiscreteHMM) -> None:
        """Initialize raw weights from an HMM via log-probability linearization.

        Computes the Jacobian of the log-transfer map at the stationary
        distribution and sets ``A``, ``B``, ``C`` accordingly. This provides
        a principled warm-start that approximates the exact filter dynamics
        near stationarity.

        Args:
            hmm (DiscreteHMM): Source HMM whose parameters define the linearization point.
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
        self.set_raw_weights(
            {
                "A_1": np.array(A1),
                "A_2": np.array(A2),
                "B": B_mat,
                "C": np.log(C_mat),
            }
        )


# ---------------------------------------------------------------------------
# ModelB  (stable linear recurrence, affine softmax readout)
# ---------------------------------------------------------------------------


class ModelB(AbstractRNN):
    """Linear RNN with stable latent dynamics and affine softmax readout.

    Recurrence: ``x_t = A x_{t-1} + B[:, y_t]``;
    readout: ``p_t = softmax(C x_t + d)``.

    ``A`` is Cayley-stable. The output bias ``d`` allows a richer readout mapping
    compared to :class:`ModelA`.
    """

    @staticmethod
    def schema(n: int, m: int) -> list[tuple[str, tuple[int,...], ConstraintType]]:
        """Return the weight schema for Model B.

        Args:
            n (int): Latent state dimensionality.
            m (int): Emission dimensionality.

        Returns:
            list[tuple[str, tuple[int], ConstraintType]]: Weight schema with entries
                for ``A`` (stable), ``B`` (unconstrained), ``C`` (unconstrained), and ``d`` (unconstrained).
        """
        return [
            ("A", (n, n), ConstraintType.STABLE),
            ("B", (n, m), ConstraintType.UNCONSTRAINED),
            ("C", (m, n), ConstraintType.UNCONSTRAINED),
            ("d", (m,), ConstraintType.UNCONSTRAINED),
        ]

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, d: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with affine softmax readout.

        Computes ``x_t = A x_{t-1} + B[:, y_t]`` and reads out via
        ``y_t = softmax(C x_t + d)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape (n, n).
            B (jax.Array): Input embedding matrix of shape (n, m).
            C (jax.Array): Readout weight matrix of shape (m, n).
            d (jax.Array): Readout bias vector of shape (m,).
            x_prev (jax.Array): Previous hidden state of shape (n,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated hidden state of shape (n,).
            y_t (jax.Array): Predicted next-emission distribution of shape (m,).
        """
        x_t = A @ x_prev + B[:, emission_t]
        y_t = jax.nn.softmax(C @ x_t + d)
        return x_t, y_t
