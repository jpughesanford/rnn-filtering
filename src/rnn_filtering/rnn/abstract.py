"""RNN architectures for sequence modelling.
"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from .parameters import _CUSTOM_CONSTRUCTOR_MAP, CONSTRUCTOR_MAP, Parameter
from .types import ConstraintType, Schema


__all__ = ["AbstractRNN"]

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
    # Training
    # ------------------------------------------------------------------


    def partition(self) -> tuple[dict, dict]:
        """Partition parameters into trainable and frozen pytrees.

        Returns:
            tuple: ``(trainable, static)`` where trainable contains unfrozen
                parameters and static contains frozen ones. Pass directly to
                :func:`rnn_filtering.train`.
        """
        return eqx.partition(self._parameters, self._is_trainable, is_leaf=lambda x: isinstance(x, Parameter))

    def combine(self, trainable: dict, static: dict) -> None:
        """Merge trainable and static pytrees back into the parameter dict.

        Args:
            trainable (dict): Trainable partition returned by :meth:`partition`.
            static (dict): Static partition returned by :meth:`partition`.
        """
        self._parameters = eqx.combine(trainable, static, is_leaf=lambda x: isinstance(x, Parameter))

    def make_forward(self, initial_condition: jax.Array) -> Callable:
        """Return a forward function closed over ``initial_condition`` for use in JIT-compiled training loops.

        Args:
            initial_condition (jax.Array): Initial latent state of shape (latent_dim,).

        Returns:
            Callable: ``(trainable, static, inputs) -> (output_states, latent_states)``
        """
        is_leaf = lambda x: isinstance(x, Parameter)

        def forward(trainable, static, inputs):
            params = eqx.combine(trainable, static, is_leaf=is_leaf)
            return self._batched_forward(params, inputs, initial_condition)

        return forward

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def respond(self, inputs: ArrayLike, initial_condition: ArrayLike | None = None) -> tuple[jax.Array, jax.Array]:
        """Run the RNN on a batch of input sequences.

        Args:
            inputs (ArrayLike): Input vectors of shape (B, T, input_dim).
                May be one-hot encodings, soft distributions, or any other
                fixed-length representation.
            initial_condition (ArrayLike, optional): Initial latent state of shape (latent_dim,).
                Defaults to zeros.

        Returns:
            output_states (jax.Array): Output states of shape (B, T, output_dim).
            latent_states (jax.Array): Latent states of shape (B, T, latent_dim).
        """
        inputs = jnp.asarray(inputs)
        initial_condition = jnp.zeros(self.latent_dim) if initial_condition is None else jnp.asarray(initial_condition)
        return self._batched_forward(self._parameters, inputs, initial_condition)

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def _batched_forward(
            cls, params: dict[str, Parameter], inputs: jax.Array, initial_condition: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """JIT-compiled batched forward pass via ``jax.vmap`` over ``jax.lax.scan``.

        Args:
            params (dict[str, Parameter]): Current RNN instance parameters.
            inputs (jax.Array): Input vectors of shape (B, T, input_dim).
            initial_condition (jax.Array): Initial hidden state of shape (latent_dim,).

        Returns:
            output_states (jax.Array): Output states of shape (B, T, output_dim).
            latent_states (jax.Array): Hidden states of shape (B, T, latent_dim).
        """
        w = {name: parameter.get_value() for name, parameter in params.items()}

        def step(x_prev, input_t):
            x_t, y_t = cls.integrate(x_prev=x_prev, input_t=input_t, **w)
            return x_t, (y_t, x_t)

        def scan_single(inputs_single):
            _, (output_states, latent_states) = jax.lax.scan(step, initial_condition, inputs_single)
            return output_states, latent_states

        return jax.vmap(scan_single)(inputs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trainable(node):
        """Returns True if a node should be differentiated during training."""
        if isinstance(node, Parameter):
            return not node.frozen
        return False

