"""RNN architectures for sequence modelling.
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn import logsumexp
from jax.typing import ArrayLike

from . import parameters
from .loss_functions import LOSS_MAP
from .parameters import _CUSTOM_CONSTRUCTOR_MAP, CONSTRUCTOR_MAP, Parameter
from .types import ConstraintType, LossType, Schema

if TYPE_CHECKING:
    from rnn_filtering.hmm.models import AbstractHMM as GenericHMM

__all__ = ["AbstractRNN", "ExactRNN", "ModelA", "ModelB", "Parameter", "parameters"]

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
# Loss resolution helper
# ---------------------------------------------------------------------------


def _resolve_loss(loss: str | Callable | None) -> Callable | None:
    """Resolve a loss specification to a callable.

    Args:
        loss: A :class:`LossType` string, a callable, or None.

    Returns:
        A callable loss function, or None.

    Raises:
        ValueError: If a string loss type is not in :data:`LOSS_MAP`. In
            particular, ``'emissions'`` is not directly usable here; pass
            one-hot vectors as ``desired_output`` with ``output_loss='kl'``,
            or use :func:`rnn_filtering.train_on_hmm` which handles embedding.
    """
    if loss is None:
        return None
    if callable(loss):
        return loss
    loss_type = LossType(loss)
    if loss_type not in LOSS_MAP:
        raise ValueError(
            f"LossType '{loss}' is not available in AbstractRNN.train. "
            "For emission-matching, supply one-hot embeddings as desired_output "
            "with output_loss='kl', or call train_on_hmm with output_loss='emissions'."
        )
    return LOSS_MAP[loss_type]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractRNN(metaclass=ABCMeta):
    """Base class for all RNN filtering models.

    The RNN maps a sequence of vector-valued inputs to a sequence of output
    distributions.  Inputs may be one-hot embeddings of discrete symbols,
    soft distributions, or any other fixed-length vector.

    Subclasses must implement:
        - ``schema(input_dim, latent_dim, output_dim)`` (staticmethod) returning parameter specifications.
        - ``integrate(..., x_prev, input_t)`` (staticmethod) defining the recurrence.

    Use :meth:`get_parameter_names`, :meth:`get_parameter_values`, and
    :meth:`set_parameter_values` to interact with the RNN parameters.

    Attributes:
        latent_dim (int): Dimensionality of the latent state.
        input_dim (int): Dimensionality of each input vector.
        output_dim (int): Dimensionality of each output vector.
        seed (int): Seed used to initialise all randomly-initialised weights.
    """

    def __init__(self, input_dim: int, latent_dim: int, output_dim: int, seed: int = 0, ic_scale: float = 0.01):
        self.input_dim: int = input_dim
        self.latent_dim: int = latent_dim
        self.output_dim: int = output_dim

        json_schema = self.schema(input_dim, latent_dim, output_dim)
        prng_key = jax.random.PRNGKey(seed)
        self.seed: int = seed

        self._parameters: dict[str, Parameter] = _instantiate_from_schema(json_schema, prng_key, ic_scale)

    @staticmethod
    @abstractmethod
    def schema(input_dim: int, latent_dim: int, output_dim: int) -> Schema:
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

        Example: if I want A to be a stable (latent x latent) matrix, and B an (output x latent) linear readout,
        then I will use:

            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {
                        "shape": (latent_dim, latent_dim),
                        "constraint": "stable"
                    },
                    "B": {
                        "shape": (output_dim, latent_dim),
                    },
                }

        In the above example, both matrices will be initialized randomly, since I did not specify their initial value.

        Args:
            input_dim (int): Input vector dimensionality.
            latent_dim (int): Latent state dimensionality.
            output_dim (int): Output vector dimensionality.

        Returns:
            Schema : parameter schema of the network
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def integrate(**kwargs):
        """Define the recurrence: ``(x_prev, input_t) -> (x_t, y_t)``.

        Subclasses should define the signature as
        ``cls.integrate(x_prev=x_prev, input_t=input_t, **W)``
        where constrained weight values from the schema are passed as keyword
        arguments of matching name.

        Args:
            x_prev (jax.Array): Previous hidden state of shape (latent_dim,).
            input_t (jax.Array): Current input vector of shape (input_dim,).
            **kwargs: Constrained weight arrays as defined by :meth:`schema`.

        Returns:
            x_t (jax.Array): Updated hidden state of shape (latent_dim,).
            y_t (jax.Array): Predicted output distribution of shape (output_dim,).
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

    def respond(self, inputs: ArrayLike, x0: ArrayLike | None = None) -> tuple[jax.Array, jax.Array]:
        """Run the RNN on a batch of input sequences.

        Args:
            inputs (ArrayLike): Input vectors of shape (B, T, input_dim).
                May be one-hot encodings, soft distributions, or any other
                fixed-length representation.
            x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,).
                Defaults to zeros.

        Returns:
            Y (jax.Array): Output states of shape (B, T, output_dim).
            X (jax.Array): Hidden states of shape (B, T, latent_dim).
        """
        inputs = jnp.asarray(inputs)
        x0 = self._resolve_x0(x0)
        return self._batched_forward(self._parameters, inputs, x0)

    def sample_loss(
        self,
        inputs: ArrayLike,
        *,
        desired_output: ArrayLike | None = None,
        desired_latent: ArrayLike | None = None,
        output_loss: str | Callable | None = None,
        latent_loss: str | Callable | None = None,
        x0: ArrayLike | None = None,
    ) -> jax.Array:
        """Run the RNN on ``inputs`` and return unaggregated per-timestep loss.

        Calls each active loss function with ``do_average=False``; custom
        callables must therefore support that keyword argument.

        Args:
            inputs (ArrayLike): Input vectors of shape (B, T, input_dim).
            desired_output (ArrayLike, optional): Target output distributions of
                shape (B, T, output_dim). Required when ``output_loss`` is set.
            desired_latent (ArrayLike, optional): Target latent states. Required
                when ``latent_loss`` is set.
            output_loss (str | Callable, optional): Output loss function.
            latent_loss (str | Callable, optional): Latent loss function.
            x0 (ArrayLike, optional): Initial hidden state. Defaults to zeros.

        Returns:
            jax.Array: Per-timestep total loss of shape (B, T).
        """
        if output_loss is None and latent_loss is None:
            raise ValueError("At least one of output_loss or latent_loss must be specified.")

        output_loss_fn = _resolve_loss(output_loss)
        latent_loss_fn = _resolve_loss(latent_loss)

        x0_vec = self._resolve_x0(x0)
        Y, X = self._batched_forward(self._parameters, jnp.asarray(inputs), x0_vec)

        total = None
        if output_loss_fn is not None:
            out = output_loss_fn(Y, jnp.asarray(desired_output), do_average=False)
            total = out
        if latent_loss_fn is not None:
            out = latent_loss_fn(X, jnp.asarray(desired_latent), do_average=False)
            total = out if total is None else total + out

        return total

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        inputs: ArrayLike,
        *,
        desired_output: ArrayLike | None = None,
        desired_latent: ArrayLike | None = None,
        output_loss: str | Callable | None = None,
        latent_loss: str | Callable | None = None,
        learning_rate: float = 2e-2,
        optimization_steps: int = 2000,
        print_every: int = 100,
        x0: ArrayLike | None = None,
    ) -> np.ndarray:
        """Train the RNN on a fixed batch of inputs.

        Only unfrozen weights are updated.  Data sampling and epoch looping are
        the caller's responsibility; see :func:`rnn_filtering.train_on_hmm` for
        a batteries-included wrapper when training on HMM data.

        Args:
            inputs (ArrayLike): Input vectors of shape (B, T, input_dim).
            desired_output (ArrayLike, optional): Target output distributions of
                shape (B, T, output_dim). Required when ``output_loss`` is set.
            desired_latent (ArrayLike, optional): Target latent states. Required
                when ``latent_loss`` is set.
            output_loss (str | Callable, optional): Loss applied to the RNN
                output.  A :class:`LossType` string or any callable with
                signature ``(result, desired) -> scalar``.  Keyword arguments
                ``clip`` and ``do_average`` are optional.
            latent_loss (str | Callable, optional): Loss applied to the RNN
                latent state.  Same callable contract as ``output_loss``.
            learning_rate (float): Adam learning rate. Defaults to 2e-2.
            optimization_steps (int): Number of gradient steps. Defaults to 2000.
            print_every (int): Print loss every this many steps (0 = silent).
                Defaults to 100.
            x0 (ArrayLike, optional): Initial hidden state. Defaults to zeros.

        Returns:
            np.ndarray: Loss history of shape (optimization_steps,).
        """
        if output_loss is None and latent_loss is None:
            raise ValueError("At least one of output_loss or latent_loss must be specified.")

        output_loss_fn = _resolve_loss(output_loss)
        latent_loss_fn = _resolve_loss(latent_loss)

        inputs_arr = jnp.asarray(inputs)
        x0_vec = self._resolve_x0(x0)

        # Dummy arrays stand in for unused desired values; they are never
        # accessed inside compute_loss because the if-branches are resolved at
        # JAX trace time (Python-level booleans).
        _dummy = jnp.zeros(0)
        desired_out = _dummy if desired_output is None else jnp.asarray(desired_output)
        desired_lat = _dummy if desired_latent is None else jnp.asarray(desired_latent)

        diff_params, static_params = eqx.partition(self._parameters, self._is_trainable, is_leaf=self._is_leaf)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(diff_params)

        batched_forward = self._batched_forward
        is_leaf = self._is_leaf

        @jax.jit
        def update_step(diff, static, state, batch_inputs, d_out, d_lat):
            def compute_loss(d, s):
                full_params = eqx.combine(d, s, is_leaf=is_leaf)
                Y, X = batched_forward(full_params, batch_inputs, x0_vec)
                loss = jnp.zeros(())
                if output_loss_fn is not None:
                    loss = loss + output_loss_fn(Y, d_out)
                if latent_loss_fn is not None:
                    loss = loss + latent_loss_fn(X, d_lat)
                return loss

            loss_val, grads = jax.value_and_grad(compute_loss)(diff, static)
            updates, next_state = optimizer.update(grads, state, diff)
            next_diff = optax.apply_updates(diff, updates)
            return next_diff, next_state, loss_val

        loss_history = np.zeros(optimization_steps)
        current_diff = diff_params

        for s in range(1, optimization_steps + 1):
            current_diff, opt_state, current_loss = update_step(
                current_diff, static_params, opt_state, inputs_arr, desired_out, desired_lat
            )
            loss_history[s - 1] = float(current_loss)
            if print_every > 0 and s % print_every == 0:
                print(f"step {s}: loss={float(current_loss):.12e}")

        self._parameters = eqx.combine(current_diff, static_params, is_leaf=self._is_leaf)
        return loss_history

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _batched_forward(
        cls, params: dict[str, Parameter], inputs: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled batched forward pass via ``jax.vmap`` over ``jax.lax.scan``.

        Args:
            params (dict[str, Parameter]): Current RNN instance parameters.
            inputs (jax.Array): Input vectors of shape (B, T, input_dim).
            x0 (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            Y (jax.Array): Output states of shape (B, T, output_dim).
            X (jax.Array): Hidden states of shape (B, T, latent_dim).
        """
        w = {name: parameter.get_value() for name, parameter in params.items()}

        def step(x_prev, input_t):
            x_t, y_t = cls.integrate(x_prev=x_prev, input_t=input_t, **w)
            return x_t, (y_t, x_t)

        def scan_single(inputs_single):
            _, (Y, X) = jax.lax.scan(step, x0, inputs_single)
            return Y, X

        return jax.vmap(scan_single)(inputs)

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
    """Exact forward-filter dynamics: ``x_t = log(A exp(x_{t-1})) + B input_t``.

    Raw weights ``A`` and ``C`` are stored in unconstrained log-space and
    softmax-normalised to column-stochastic matrices before the scan.
    """

    @staticmethod
    def schema(input_dim: int, latent_dim: int, output_dim: int) -> Schema:
        if input_dim != output_dim:
            raise ValueError(f"ExactRNN requires input_dim == output_dim, got {input_dim} != {output_dim}.")
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
            "B": {"shape": (latent_dim, input_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (output_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, input_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Exact nonlinear forward-filter recurrence.

        Computes ``x_t = log(A exp(x_{t-1})) + B input_t`` (normalised in
        log-space) and reads out via ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Column-stochastic transfer matrix of shape (latent_dim, latent_dim).
            B (jax.Array): Input projection matrix of shape (latent_dim, input_dim).
            C (jax.Array): Column-stochastic readout matrix of shape (output_dim, latent_dim).
            x_prev (jax.Array): Previous log-posterior of shape (latent_dim,).
            input_t (jax.Array): Current input vector of shape (input_dim,).

        Returns:
            x_t (jax.Array): Updated log-posterior of shape (latent_dim,).
            y_t (jax.Array): Predicted output distribution of shape (output_dim,).
        """
        x_t = jnp.log(A @ jnp.exp(x_prev)) + B @ input_t
        x_t = x_t - logsumexp(x_t)
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_weights(self, hmm: GenericHMM) -> None:
        """Set parameter values to match the HMM exactly.

        This produces an RNN that exactly reproduces the AbstractHMM forward
        filter.

        Args:
            hmm (AbstractHMM): Source HMM whose parameters are copied.
        """
        transfer_matrix = np.array(jax.vmap(hmm.transfer_operator.apply)(jnp.eye(hmm.latent_dim))).T
        emission_matrix = np.array(jax.vmap(hmm.emission_operator.apply)(jnp.eye(hmm.latent_dim))).T
        self.set_parameter_values(
            {
                "A": transfer_matrix,
                "B": np.log(emission_matrix.T),
                "C": emission_matrix @ transfer_matrix,
            }
        )


# ---------------------------------------------------------------------------
# Model A  (stable linear recurrence, stochastic linear of softmax readout. See paper for more. )
# ---------------------------------------------------------------------------


class ModelA(AbstractRNN):
    """Linear RNN with stable latent dynamics and stochastic readout.

    Recurrence: ``x_t = A x_{t-1} + B input_t``;
    readout: ``p_t = C softmax(x_t)``.

    ``A`` is parameterised via the Cayley transform to guarantee spectral
    radius <= 1.  ``C`` is softmax-normalised to a column-stochastic matrix.
    """

    @staticmethod
    def schema(input_dim: int, latent_dim: int, output_dim: int) -> Schema:
        if input_dim != output_dim:
            raise ValueError(f"ModelA requires input_dim == output_dim, got {input_dim} != {output_dim}.")
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STABLE},
            "B": {"shape": (latent_dim, input_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (output_dim, latent_dim), "constraint": ConstraintType.STOCHASTIC},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, x_prev: jax.Array, input_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with stochastic readout.

        Computes ``x_t = A x_{t-1} + B input_t`` and reads out via
        ``y_t = C softmax(x_t)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape (latent_dim, latent_dim).
            B (jax.Array): Input projection matrix of shape (latent_dim, input_dim).
            C (jax.Array): Column-stochastic readout matrix of shape (output_dim, latent_dim).
            x_prev (jax.Array): Previous hidden state of shape (latent_dim,).
            input_t (jax.Array): Current input vector of shape (input_dim,).

        Returns:
            x_t (jax.Array): Updated hidden state of shape (latent_dim,).
            y_t (jax.Array): Predicted output distribution of shape (output_dim,).
        """
        x_t = A @ x_prev + B @ input_t
        y_t = C @ jax.nn.softmax(x_t)
        return x_t, y_t

    def initialize_astar(self, hmm: GenericHMM) -> None:
        """Initialize parameter values from an AbstractHMM via log-probability linearization.

        Computes the Jacobian of the log-transfer map at the stationary
        distribution and sets ``A``, ``B``, ``C`` accordingly. This provides
        a principled warm-start that approximates the exact filter dynamics
        near stationarity.

        Args:
            hmm (AbstractHMM): Source HMM whose parameters define the linearization point.
        """
        transfer_matrix = np.array(jax.vmap(hmm.transfer_operator.apply)(jnp.eye(hmm.latent_dim))).T
        emission_matrix = np.array(jax.vmap(hmm.emission_operator.apply)(jnp.eye(hmm.latent_dim))).T
        p = hmm.latent_stationary_density

        Tp = transfer_matrix @ p
        J = (transfer_matrix * p[None, :]) / Tp[:, None]
        f = np.log(transfer_matrix @ p)
        bias = f - J @ np.log(p)

        self.set_parameter_values(
            {"A": J, "B": np.log(emission_matrix.T) + bias[:, None], "C": emission_matrix @ transfer_matrix}
        )


# ---------------------------------------------------------------------------
# Model B  (stable linear recurrence, softmax of affine readout. See paper for more. )
# ---------------------------------------------------------------------------


class ModelB(AbstractRNN):
    """Linear RNN with stable latent dynamics and affine softmax readout.

    Recurrence: ``x_t = A x_{t-1} + B input_t``;
    readout: ``p_t = softmax(C x_t + d)``.

    ``A`` is Cayley-stable. The output bias ``d`` allows a richer readout mapping
    compared to :class:`ModelA`.
    """

    @staticmethod
    def schema(input_dim: int, latent_dim: int, output_dim: int) -> Schema:
        if input_dim != output_dim:
            raise ValueError(f"ModelB requires input_dim == output_dim, got {input_dim} != {output_dim}.")
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": ConstraintType.STABLE},
            "B": {"shape": (latent_dim, input_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "C": {"shape": (output_dim, latent_dim), "constraint": ConstraintType.UNCONSTRAINED},
            "d": {"shape": (output_dim,), "constraint": ConstraintType.UNCONSTRAINED},
        }

    @staticmethod
    def integrate(
        A: jax.Array, B: jax.Array, C: jax.Array, d: jax.Array, x_prev: jax.Array, input_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Linear recurrence with affine softmax readout.

        Computes ``x_t = A x_{t-1} + B input_t`` and reads out via
        ``y_t = softmax(C x_t + d)``.

        Args:
            A (jax.Array): Stable dynamics matrix of shape (latent_dim, latent_dim).
            B (jax.Array): Input projection matrix of shape (latent_dim, input_dim).
            C (jax.Array): Readout weight matrix of shape (output_dim, latent_dim).
            d (jax.Array): Readout bias vector of shape (output_dim,).
            x_prev (jax.Array): Previous hidden state of shape (latent_dim,).
            input_t (jax.Array): Current input vector of shape (input_dim,).

        Returns:
            x_t (jax.Array): Updated hidden state of shape (latent_dim,).
            y_t (jax.Array): Predicted output distribution of shape (output_dim,).
        """
        x_t = A @ x_prev + B @ input_t
        y_t = jax.nn.softmax(C @ x_t + d)
        return x_t, y_t
