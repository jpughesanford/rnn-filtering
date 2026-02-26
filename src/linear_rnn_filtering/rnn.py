"""RNN architectures for approximating HMM forward filtering.

Each model takes a sequence of discrete emissions and produces a
predicted next-token probability distribution at every time step.

"""

import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn import logsumexp
from jax.typing import ArrayLike

from . import parameters
from .hmm import DiscreteHMM
from .loss_functions import LOSS_MAP
from .parameters import _CUSTOM_CONSTRUCTOR_MAP, CONSTRUCTOR_MAP, Parameter
from .types import ConstraintType, LossType, Schema

__all__ = ["AbstractRNN", "ExactRNN", "ModelA", "ModelB", "Parameter","parameters"]

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _instantiate_from_schema(schema: Schema, prng_key: jax.Array, ic_scale: float = 0.01) -> dict[str, Parameter]:
    """A valid schema for RNN architecture has the format:
    `schema = {
        "param_name": {
            "shape": tuple[int,...], # defaults to (1, )
            "constraint": str | ConstraintType, # defaults to "unconstrained"
            "initial_value": jax.ArrayLike # defaults to random initialization
        },
        "next_param_name": ...,
        ...
    }`

    This method takes a schema as input, validates it, and returns a dict of initialized :class:`Parameter` instances
    for the RNN to use. All fields (shape, constraint, initial_value) are optional.

    Raises:
        ValueError: whenever schema element value is invalid. Error message will detail the specific value issue.
        TypeError: whenever schema structure is invalid. Error message will detail the specific type issue.

    Warns:
        UserWarning: If fields are provided that are not used

    Returns:
        dict[str,Parameter]: a dict of valid Parameter instances, using the same parameter names as were passed in.
    """

    if isinstance(schema, dict):
        result: dict[str, Parameter] = {}
        keys = jax.random.split(prng_key, len(schema))

        for field, key in zip(schema, keys, strict=True):
            entry = schema[field]
            if not isinstance(field, str):
                raise ValueError("schema fields must be string parameter names.")
            if not isinstance(entry, dict):
                raise ValueError("schema entries must be (potentially empty) dict objects.")
            unexpected = entry.keys() - {"shape", "initial_value", "constraint"}
            if unexpected:
                warnings.warn(
                    f"Unexpected parameter names in schema will be unused: {unexpected}.", UserWarning, stacklevel=3
                )
            shape = entry.get("shape", (1,))
            constraint_str = entry.get("constraint", "unconstrained")
            if constraint_str in _CUSTOM_CONSTRUCTOR_MAP:
                p = _CUSTOM_CONSTRUCTOR_MAP[constraint_str](shape)
            else:
                p = CONSTRUCTOR_MAP[ConstraintType(constraint_str)](shape)
            result[field] = (
                p.set_value(entry["initial_value"]) if "initial_value" in entry else p.randomize_dof(key, ic_scale)
            )  # both set_value and randomize_dof return a new parameter instance

        return result

    raise TypeError("schema must be a dictionary.")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractRNN(metaclass=ABCMeta):
    """Base class for all RNN filtering models.

    Subclasses must implement:
        - ``schema(latent_dim, emission_dim)`` (staticmethod) returning parameter specifications.
        - ``integrate(..., x_prev, emission_t)`` (staticmethod) defining the recurrence.

    Use :meth:`get_parameter_names`, :meth:`get_parameter_values`, and :meth:`set_parameter_values` to interact with
    the RNN parameters.

    Attributes:
        latent_dim (int): Dimensionality of the latent state.
        emission_dim (int): Number of discrete emissions.
        seed (int): the seed used to initialize all weights (whose initial value is not specified in the schema).
    """

    def __init__(self, latent_dim: int, emission_dim: int, seed: int = 0, ic_scale: float = 0.01):
        self.latent_dim: int = latent_dim
        self.emission_dim: int = emission_dim

        json_schema = self.schema(latent_dim, emission_dim)
        prng_key = jax.random.PRNGKey(seed)
        self.seed: int = seed

        self._parameters: dict[str, Parameter] = _instantiate_from_schema(json_schema, prng_key, ic_scale)

    @staticmethod
    @abstractmethod
    def schema(latent_dim: int, emission_dim: int) -> Schema:
        """Return the parameter schema for this architecture. A valid schema for RNN architecture has the format:
            schema = {
                "param_name": {
                    "shape": tuple[int,...], # defaults to (1, )
                    "constraint": str | ConstraintType, # defaults to "unconstrained"
                    "initial_value": jax.ArrayLike # defaults to random initialization
                },
                "next_param_name": ...,
                ...
            }

        All fields (shape, constraint, initial_value) are optional.

        Example: if I want A to be a stable (latent x latent) matrix, and B an (emission x latent) linear readout,
        then I will use:

            @staticmethod
            def schema(latent_dim, emission_dim):
                return {
                    "A": {
                        "shape": (latent_dim, latent_dim),
                        "constraint": "stable"
                    },
                    "B": {
                        "shape": (emission_dim, latent_dim),
                    },
                }

        In the above example, both matrices will be initialized randomly, since I did not specify their initial value.

        Args:
            latent_dim (int): Latent state dimensionality.
            emission_dim (int): Emission dimensionality.

        Returns:
            Schema : parameter schema of the network
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def integrate(**kwargs):
        """Define the recurrence: ``(x_prev, emission_t) -> (x_t, y_t)``.

        Subclasses should define the signature as
        ``cls.integrate(x_prev=x_prev, emission_t=emission_t, **W)``
        where values of the schema entries are passed in as keyword arguments of matching name.

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
    # Parameter management (getting/setting/freezing)
    # ------------------------------------------------------------------

    def _warn_unknown_keys(self, keys: Iterable[str], caller: str, stacklevel: int = 3) -> None:
        unknown = set(keys) - set(self._parameters.keys())
        if unknown:
            warnings.warn(f"{caller}: unknown parameter names {unknown}", stacklevel=stacklevel)

    def freeze(self, parameter_names: Iterable[str]) -> None:
        """Prevent listed parameters from being updated during training.

        Warns:
            UserWarning: If unknown keys are provided
        """
        self._warn_unknown_keys(parameter_names, "freeze")
        for key in self._parameters.keys() & parameter_names:
            self._parameters[key] = self._parameters[key].freeze()

    def unfreeze(self, parameter_names: Iterable[str]) -> None:
        """Allow listed parameters to be updated during training.

        Warns:
            UserWarning: If unknown keys are provided
        """
        self._warn_unknown_keys(parameter_names, "unfreeze")
        for key in self._parameters.keys() & parameter_names:
            self._parameters[key] = self._parameters[key].unfreeze()

    def set_parameter_values(self, new_values: dict[str, ArrayLike]) -> None:
        """Overwrite parameter values by name. Sets the constrained value, not the internal degrees of freedom.

        Warns:
            UserWarning: If unknown keys are provided
        """
        self._warn_unknown_keys(new_values.keys(), "set_parameter_values")
        for key in self._parameters.keys() & new_values.keys():
            self._parameters[key] = self._parameters[key].set_value(new_values[key])

    def get_parameter_values(self, parameter_names: Iterable[str]) -> dict[str, ArrayLike]:
        """Retrieve parameter values by name. Returns the constrained value, not the internal degrees of freedom.

        Warns:
            UserWarning: If unknown keys are provided
        """
        result = {}
        self._warn_unknown_keys(parameter_names, "get_parameter_values")
        for key in self._parameters.keys() & parameter_names:
            result[key] = self._parameters[key].get_value()
        return result

    def get_parameter_names(self) -> set[str]:
        return set(self._parameters.keys())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, emissions: ArrayLike, x0: ArrayLike | None = None) -> tuple[jax.Array, jax.Array]:
        """Predict next-token distributions for a batch of emission sequences.

        Args:
            emissions (ArrayLike): Observed emission indices of shape (B, T).
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            Y (jax.Array): Output of the RNN, of shape (B, T, emission_dim).
            X (jax.Array): Latent state of the RNN, of shape (B, T, latent_dim).
        """
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        x0 = self._resolve_x0(x0)
        output, latent = self._batched_forward(self._parameters, emissions, x0)
        return jnp.asarray(output), jnp.asarray(latent)

    def sample_loss(
        self,
        hmm: DiscreteHMM,
        *,
        loss: str = LossType.KL,
        batch_size: int = 100,
        time_steps: int = 1000,
        x0: ArrayLike | None = None,
    ) -> jax.Array:
        """Samples a new batch of trajectories from an :class:``DiscreteHMM`` and returns the
        unaggregated loss over every time step and sample

        Args:
            hmm (DiscreteHMM): HMM instance used to generate evaluation data.
            loss (str, optional): string describing the loss function to use. Currently accepted values are
                "emissions", "kl" (default), and "hilbert". See :class:`LossType` for details.
            batch_size (int, optional): Number of independent trajectories to sample. Defaults to 100.
            time_steps (int, optional): Length of each sampled trajectory. Defaults to 1000.
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,). Defaults to zeros.

        Returns:
            jax.Array: Per-timestep loss values of shape (B, T) or (B, T-1) depending on loss type.
        """
        loss = LossType(loss)
        loss_fn, needs_posterior = LOSS_MAP[loss]
        _, emissions = hmm.sample(batch_size, time_steps)
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        x0_vec = self._resolve_x0(x0)
        if needs_posterior:
            _, next_token_posterior = hmm.compute_posterior(emissions)
        else:
            next_token_posterior = None
        return loss_fn(
            self._batched_forward, self._parameters, emissions, next_token_posterior, x0_vec, do_average=False
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        hmm: DiscreteHMM,
        *,
        loss: str = LossType.KL,
        batch_size: int = 100,
        time_steps: int = 1000,
        num_epochs: int = 1,
        learning_rate: float = 2e-2,
        optimization_steps: int = 2000,
        print_every: int = 100,
        x0: ArrayLike | None = None,
    ) -> ArrayLike:
        """Batch trains the RNN using sequences sampled from a passed in :class:`DiscreteHMM` instance.

        Handles data sampling, optimizer construction, and gradient updates. Only unfrozen weights are trained.

        Args:
            hmm (DiscreteHMM): HMM instance used to generate training data.
            loss (str, optional): string describing the loss function to use. Currently accepted values are
                "emissions", "kl" (default), and "hilbert". See :class:`LossType` for details.
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
        loss = LossType(loss)
        loss_fn, needs_posterior = LOSS_MAP[loss]
        batched_forward = self._batched_forward

        # partition into differentiable and static parts
        diff_params, static_params = eqx.partition(self._parameters, self._is_trainable, is_leaf=self._is_leaf)

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(diff_params)

        @jax.jit
        def update_step(the_differentiable_params, the_static_params, the_state, the_emissions, the_targets):
            def compute_loss(diff, static):
                full_params = eqx.combine(diff, static, is_leaf=self._is_leaf)
                return loss_fn(batched_forward, full_params, the_emissions, the_targets, x0_vec)

            loss_val, grads = jax.value_and_grad(compute_loss)(the_differentiable_params, the_static_params)
            updates, next_state = optimizer.update(grads, the_state, the_differentiable_params)
            next_diff_params = optax.apply_updates(the_differentiable_params, updates)

            return next_diff_params, next_state, loss_val

        # training loop
        current_diff = diff_params
        for epoch in range(num_epochs):
            _, emissions = hmm.sample(batch_size, time_steps)
            emissions = jnp.asarray(emissions, dtype=jnp.int32)
            if needs_posterior:
                _, next_token_posterior = hmm.compute_posterior(emissions)
            else:
                next_token_posterior = None

            for s in range(1, optimization_steps + 1):
                current_diff, opt_state, current_loss = update_step(
                    current_diff, static_params, opt_state, emissions, next_token_posterior
                )

                loss_history[s - 1, epoch] = current_loss

                if s % print_every == 0:
                    print(f"epoch {epoch}, step {s}: loss={float(current_loss):.12e}")

        self._parameters = eqx.combine(current_diff, static_params, is_leaf=self._is_leaf)
        return loss_history

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _forward_scan(
        cls, params: dict[str, Parameter], emissions: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Run the RNN on a single emission sequence via ``jax.lax.scan``.

        Args:
            params (dict[str, Parameters]): current RNN instance parameters.
            emissions (jax.Array): Emission indices of shape (T,), dtype int32.
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            Y (jax.Array): Predicted distributions of shape (T, emission_dim).
            X (jax.Array): Hidden states of shape (T, latent_dim).
        """
        w = {name: parameter.get_value() for name, parameter in params.items()}

        def step(x_prev, emission_t):
            x_t, y_t = cls.integrate(x_prev=x_prev, emission_t=emission_t, **w)
            return x_t, (y_t, x_t)

        _, (Y, X) = jax.lax.scan(step, x0, emissions)
        return Y, X

    @classmethod
    def _batched_forward(
        cls, params: dict[str, Parameter], emissions: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Vectorised forward pass over a batch of sequences via ``jax.vmap``.

        Args:
            params (dict[str,  Parameter]): Dictionary of RNN parameters.
            emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            Y (jax.Array): Predicted distributions of shape (B, T, emission_dim).
            X (jax.Array): Hidden states of shape (B, T, latent_dim).
        """
        return jax.vmap(cls._forward_scan, in_axes=(None, 0, None))(params, emissions, x0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_x0(self, x0: ArrayLike | None) -> jax.Array:
        return jnp.zeros(self.latent_dim) if x0 is None else jnp.asarray(x0)

    @staticmethod
    def _is_trainable(node):
        """Returns True if a node should be differentiated during training."""
        if isinstance(node, Parameter):
            return not node.frozen
        return False

    @staticmethod
    def _is_leaf(x):
        """Used by JAX to tell if a value in the parameter dict is a leaf.
        At the moment, the architecture is set up such that _parameters is a flat dict of
        modules. All values are Parameters and all values are leaves."""
        return isinstance(x, Parameter)


# ---------------------------------------------------------------------------
# ExactRNN  (ground-truth nonlinear filter. See paper for more. )
# ---------------------------------------------------------------------------


class ExactRNN(AbstractRNN):
    """Exact forward-filter dynamics: ``x_t = log(A exp(x_{t-1})) + B[:, y_t]``.

    Raw weights ``A`` and ``C`` are stored in unconstrained log-space and
    softmax-normalised to column-stochastic matrices before the scan.
    """

    @staticmethod
    def schema(latent_dim: int, emission_dim: int) -> Schema:
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
            "B": {"shape": (latent_dim, emission_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (emission_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Exact nonlinear forward-filter recurrence.

        Computes ``x_t = log(A exp(x_{t-1})) + B[:, y_t]`` (normalised in
        log-space) and reads out via ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Column-stochastic transfer matrix of shape ( latent_dim,  latent_dim).
            B (jax.Array): Log-emission matrix of shape ( latent_dim, emission_dim).
            C (jax.Array): Column-stochastic readout matrix of shape (emission_dim,  latent_dim).
            x_prev (jax.Array): Previous log-posterior of shape ( latent_dim,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated log-posterior of shape ( latent_dim,).
            y_t (jax.Array): Predicted next-emission distribution of shape (emission_dim,).
        """
        x_t = jnp.log(A @ jnp.exp(x_prev)) + B[:, emission_t]
        x_t = x_t - logsumexp(x_t)
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_weights(self, hmm: DiscreteHMM) -> None:
        """Set parameter values to match the HMM exactly.

        This produces an RNN that exactly reproduces the HMM forward
        filter.

        Args:
            hmm (DiscreteHMM): Source HMM whose parameters are copied.
        """
        self.set_parameter_values(
            {
                "A": hmm.transfer_matrix,
                "B": np.log(hmm.emission_matrix.T),
                "C": hmm.emission_matrix @ hmm.transfer_matrix,
            }
        )


# ---------------------------------------------------------------------------
# Model A  (stable linear recurrence, stochastic linear of softmax readout. See paper for more. )
# ---------------------------------------------------------------------------


class ModelA(AbstractRNN):
    """Linear RNN with stable latent dynamics and stochastic readout.

    Recurrence: ``x_t = A x_{t-1} + B[:, y_t]``;
    readout: ``p_t = C softmax(x_t)``.

    ``A`` is parameterised via the Cayley transform to guarantee spectral
    radius <= 1.  ``C`` is softmax-normalised to a column-stochastic matrix.
    """

    @staticmethod
    def schema(latent_dim: int, emission_dim: int) -> Schema:
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STABLE},
            "B": {"shape": (latent_dim, emission_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (emission_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with stochastic readout.

        Computes ``x_t = A x_{t-1} + B[:, y_t]`` and reads out via
        ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape ( latent_dim,  latent_dim).
            B (jax.Array): Input embedding matrix of shape ( latent_dim, emission_dim).
            C (jax.Array): Column-stochastic readout matrix of shape (emission_dim,  latent_dim).
            x_prev (jax.Array): Previous hidden state of shape ( latent_dim,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated hidden state of shape ( latent_dim,).
            y_t (jax.Array): Predicted next-emission distribution of shape (emission_dim,).
        """
        x_t = A @ x_prev + B[:, emission_t]
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_astar(self, hmm: DiscreteHMM) -> None:
        """Initialize parameter values from an HMM via log-probability linearization.

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

        self.set_parameter_values({"A": J, "B": np.log(E.T) + bias[:, None], "C": E @ T})


# ---------------------------------------------------------------------------
# Model B  (stable linear recurrence, softmax of affine readout. See paper for more. )
# ---------------------------------------------------------------------------


class ModelB(AbstractRNN):
    """Linear RNN with stable latent dynamics and affine softmax readout.

    Recurrence: ``x_t = A x_{t-1} + B[:, y_t]``;
    readout: ``p_t = softmax(C x_t + d)``.

    ``A`` is Cayley-stable. The output bias ``d`` allows a richer readout mapping
    compared to :class:`ModelA`.
    """

    @staticmethod
    def schema(latent_dim: int, emission_dim: int) -> Schema:
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STABLE},
            "B": {"shape": (latent_dim, emission_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (emission_dim, latent_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "d": {"shape": (emission_dim,), "constraint": ConstraintType.UNCONSTRAINED},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, d: jax.Array, x_prev: jax.Array, emission_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with affine softmax readout.

        Computes ``x_t = A x_{t-1} + B[:, y_t]`` and reads out via
        ``y_t = softmax(C x_t + d)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape ( latent_dim,  latent_dim).
            B (jax.Array): Input embedding matrix of shape ( latent_dim, emission_dim).
            C (jax.Array): Readout weight matrix of shape (emission_dim,  latent_dim).
            d (jax.Array): Readout bias vector of shape (emission_dim,).
            x_prev (jax.Array): Previous hidden state of shape ( latent_dim,).
            emission_t (jax.Array): Current emission index (scalar int32).

        Returns:
            x_t (jax.Array): Updated hidden state of shape ( latent_dim,).
            y_t (jax.Array): Predicted next-emission distribution of shape (emission_dim,).
        """
        x_t = A @ x_prev + B[:, emission_t]
        y_t = jax.nn.softmax(C @ x_t + d)
        return x_t, y_t
