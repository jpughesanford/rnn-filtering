from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .types import ConstraintType

__all__ = ["CONSTRUCTOR_MAP", "_CUSTOM_CONSTRUCTOR_MAP", "Parameter", "register_parameter_type"]

# ---------------------------------------------------------------------------
# Parameter base (unconstrained)
# ---------------------------------------------------------------------------


class Parameter(eqx.Module):
    """A default parameter of the RNN. A parameter is any object that can be parameterized by a
     Jax array.

    The purpose of this class is to allow the user to define constrained weights for the network. For example, in
    linear RNNs, you often want the recurrent weight matrix to be a stable matrix. Making the weight matrix a
    StableParameter allows the user to enforce that object remains stable over training.

    To implement this, every Parameter has an internal `dof` field, representing the degrees of freedom that JAX is
    allowed to vary, and implements a `get_value()` function the user can call to read the parameters constrained value.

    This Parameter class defaults to `dof` being equal to the Parameters value, and should be thought of as an
    unconstrained parameter. Subclasses of this class implement further constraints. For these constrained parameters,
    `dof` will be different from the Parameters value.

    Since JAX requires immutability, this class extends the equinox Module class.

    Attributes:
        dof (jax.Array):
            the degrees of freedom of the parameter. These are what will be optimized by jax.
            The parameter value must be an invertible function of these dof.
        shape (tuple[int,...]):
            the shape of the parameter.
        frozen (bool): if true, JAX will not update this parameter during training.
    """

    dof: jax.Array | tuple[jax.Array, ...]
    shape: tuple[int, ...] = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], frozen: bool = False) -> None:
        self.shape = shape
        self.frozen = frozen
        shapes = self._dof_shapes()
        if len(shapes) == 1:
            self.dof = jnp.zeros(shapes[0])
        else:
            self.dof = tuple(jnp.zeros(s) for s in shapes)

    def _dof_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Given the shape of the value (self.shape), this method returns a tuples of shapes of all internal dof
        for this parameter.

        Returns:
            shapes (tuple[tuple[int,...],...]): a tuple of shapes describing the internal dof for this parameter.
        """
        return (self.shape,)

    def freeze(self) -> "Parameter":
        """Freezes this parameter so that its degrees of freedom are not varied during training.
        Because JAX requires immutability, this method returns a new Parameter object.
        Returns:
            Parameter: the new parameter object.
        """
        new_instance = type(self)(self.shape, frozen=True)
        return eqx.tree_at(lambda s: s.dof, new_instance, self.dof)

    def unfreeze(self) -> "Parameter":
        """Unfreezes this parameter so that its degrees of freedom are varied during training.
        Because JAX requires immutability, this method returns a new Parameter object.
        Returns:
            Parameter: the new parameter object.
        """
        new_instance = type(self)(self.shape, frozen=False)
        return eqx.tree_at(lambda s: s.dof, new_instance, self.dof)

    def randomize_dof(self, prng_key: jax.Array, ic_scale: float = 0.01) -> "Parameter":
        value = jax.random.normal(prng_key, self.shape) * ic_scale
        return self.set_value(value)

    def get_value(self) -> jax.Array:
        return self.dof

    def set_value(self, value: ArrayLike) -> "Parameter":
        value = jnp.asarray(value)
        if value.shape != self.shape:
            raise ValueError(f"Value shape {value.shape} does not match parameter shape {self.shape}.")
        return eqx.tree_at(lambda s: s.dof, self, value)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


class StochasticParameter(Parameter):
    """Parametrization of a stochastic ndarray. Jax optimizes X, and the parameter evaluates
    to Y = softmax(X,axis=0).
    """

    def get_value(self) -> jax.Array:
        return jax.nn.softmax(self.dof, axis=0)

    def set_value(self, value: ArrayLike) -> "StochasticParameter":
        value = jnp.asarray(value)
        if jnp.any(value < 0):
            raise ValueError("Stochastic parameter value must be non-negative.")
        if value.shape != self.shape:
            raise ValueError(f"Value shape {value.shape} does not match parameter shape {self.shape}.")
        value = jnp.log(value)
        return eqx.tree_at(lambda s: s.dof, self, value)

    def randomize_dof(self, prng_key: jax.Array, ic_scale: float = 0.01) -> "Parameter":
        value = jax.random.uniform(prng_key, self.shape) * ic_scale
        return self.set_value(value)


# ---------------------------------------------------------------------------
# Nonnegative
# ---------------------------------------------------------------------------


class NonnegativeParameter(Parameter):
    """Parametrization of a non-negative ndarray. Jax optimizes X, and the parameter evaluates to Y = X*X."""

    def get_value(self) -> jax.Array:
        return self.dof**2

    def set_value(self, value: ArrayLike) -> "NonnegativeParameter":
        value = jnp.asarray(value)
        if value.shape != self.shape:
            raise ValueError(f"Value shape {value.shape} does not match parameter shape {self.shape}.")
        value = jnp.sqrt(jnp.abs(value))
        return eqx.tree_at(lambda s: s.dof, self, value)


# ---------------------------------------------------------------------------
# Stable
# ---------------------------------------------------------------------------


class StableParameter(Parameter):
    """Parameterization of a square matrix constrained to have spectral radius <= 1 (Schur stable).

    Internally stores two unconstrained arrays as degrees of freedom:
        - ``A1``: shape (n, n), used to build a positive-definite component via ``A1 @ A1.T + eps * I``.
        - ``A2``: shape (n*(n-1)//2,), the lower-triangular entries of a skew-symmetric component.

    ``get_value()`` returns the Cayley transform of their sum, which is guaranteed to have all
    eigenvalues strictly inside the unit circle. Matrices with eigenvalue exactly -1 are not
    representable by this parameterization.
    """

    def __init__(self, shape: tuple[int, ...], frozen: bool = False) -> None:
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"StableParameter requires a square matrix shape, got {shape}.")
        super().__init__(shape, frozen)

    def _dof_shapes(self) -> tuple[tuple[int, ...], ...]:
        n = self.shape[0]
        return (n, n), (n * (n - 1) // 2,)

    def randomize_dof(self, prng_key: jax.Array, ic_scale: float = 0.01) -> "StableParameter":
        shapes = self._dof_shapes()
        keys = jax.random.split(prng_key, len(shapes))
        value = tuple(jax.random.normal(key, shape) * ic_scale for key, shape in zip(keys, shapes, strict=True))
        return eqx.tree_at(lambda s: s.dof, self, value)

    def get_value(self) -> jax.Array:
        return StableParameter.params_to_stable_matrix(*self.dof)

    def set_value(self, value: ArrayLike) -> "StableParameter":
        value = jnp.asarray(value)
        if value.shape != self.shape:
            raise ValueError(f"Value shape {value.shape} does not match parameter shape {self.shape}.")
        eigenvalues = jnp.linalg.eigvals(value)
        if jnp.max(jnp.abs(eigenvalues)) > 1:
            raise ValueError("Stable parameter value must be a stable matrix.")
        value = StableParameter.stable_matrix_to_params(value)
        return eqx.tree_at(lambda s: s.dof, self, value)

    @staticmethod
    def cayley(A: jax.Array) -> jax.Array:
        """Cayley transform: maps a square matrix to an orthogonal matrix.

        Computes ``(I - A)^{-T} (I + A)^T``.

        Args:
            A (jax.Array): Square matrix of shape (n, n).

        Returns:
            jax.Array: Orthogonal matrix of shape (n, n).
        """
        eye = jnp.eye(A.shape[0])
        return jnp.linalg.solve((eye + A).T, (eye - A).T).T

    @staticmethod
    def params_to_stable_matrix(A1: jax.Array, A2: jax.Array, epsilon: float = 1e-6) -> jax.Array:
        """Map unconstrained parameters to a matrix with spectral radius <= 1.

        Uses a Cayley-transform parameterization:
          1. ``A1 A1^T + eps I`` gives a symmetric positive-definite component.
          2. ``A2`` (lower-triangular entries) gives a skew-symmetric component.
          3. The Cayley transform of their sum is guaranteed to have all
             eigenvalues on or inside the unit circle.

        Args:
            A1 (jax.Array): Free parameters for the positive-definite part, shape (n, n).
            A2 (jax.Array): Free parameters for the skew-symmetric part, shape (n*(n-1)/2,).
            epsilon (float, optional): Regularization for positive definiteness. Defaults to 1e-6.

        Returns:
            A_stable (jax.Array): Matrix with spectral radius <= 1, shape (n, n).
        """
        n = A1.shape[0]
        eye = jnp.eye(n)
        A_pd = A1 @ A1.T + epsilon * eye

        A_lower = jnp.zeros((n, n))
        idx = jnp.tril_indices(n, k=-1)
        A_lower = A_lower.at[idx].set(A2)
        A_skew = 0.5 * (A_lower - A_lower.T)

        return StableParameter.cayley(A_pd + A_skew)

    @staticmethod
    def stable_matrix_to_params(A_stable: jax.Array, epsilon: float = 1e-6) -> tuple[jax.Array, jax.Array]:
        """Invert :func:`params_to_stable_matrix`: recover ``(A1, A2)`` from a stable matrix.

        Args:
            A_stable (jax.Array): Matrix with spectral radius <= 1, shape (n, n).
            epsilon (float, optional): Same regularization used in the forward transform. Defaults to 1e-6.

        Returns:
            A1 (jax.Array): Cholesky factor of the positive-definite component, shape (n, n).
            A2 (jax.Array): Lower-triangular entries of the skew-symmetric component, shape (n*(n-1)/2,).
        """
        n = A_stable.shape[0]
        eye = jnp.eye(n)

        eigenvalues = np.linalg.eigvals(np.array(A_stable))
        if np.any(np.abs(eigenvalues + 1) < 1e-8):
            raise ValueError(
                "stable_matrix_to_params: matrices with eigenvalue -1 are not representable "
                "by the Cayley parameterization."
            )

        A_sum = StableParameter.cayley(A_stable)

        A_pd = 0.5 * (A_sum + A_sum.T)
        SPD = A_pd - epsilon * eye
        vals, vecs = jnp.linalg.eigh(SPD)
        vals = jnp.maximum(vals, epsilon)  # epsilon >> float32 machine eps, so Cholesky stays stable
        SPD = (vecs * vals) @ vecs.T
        A1 = jnp.linalg.cholesky(SPD)

        A_skew = 0.5 * (A_sum - A_sum.T)
        idx = jnp.tril_indices(n, k=-1)
        A2 = 2 * A_skew[idx]

        return A1, A2


# ---------------------------------------------------------------------------
# Constructor map. Used for initializing parameters as CONSTRUCTOR_MAP[ConstraintType](args)
# ---------------------------------------------------------------------------

CONSTRUCTOR_MAP: dict[ConstraintType, Callable] = {
    ConstraintType.UNCONSTRAINED: Parameter,
    ConstraintType.STABLE: StableParameter,
    ConstraintType.STOCHASTIC: StochasticParameter,
    ConstraintType.NONNEGATIVE: NonnegativeParameter,
}

_CUSTOM_CONSTRUCTOR_MAP: dict[str, type[Parameter]] = {}


def register_parameter_type(name: str, cls: type[Parameter]) -> None:
    """Register a custom :class:`Parameter` subclass under a string name.

    Once registered, ``name`` can be used as the ``"constraint"`` field in any
    schema dict, exactly like the built-in values ``"stable"``, ``"stochastic"``, etc.

    Args:
        name: The string key to use in schema dicts (e.g. ``"orthogonal"``).
        cls: A :class:`Parameter` subclass to instantiate for that constraint.

    Raises:
        TypeError: If ``cls`` is not a subclass of :class:`Parameter`.
        ValueError: If ``name`` conflicts with a built-in :class:`ConstraintType` value.
    """
    if not (isinstance(cls, type) and issubclass(cls, Parameter)):
        raise TypeError(f"cls must be a subclass of Parameter, got {cls!r}.")
    built_in_names = {member.value for member in ConstraintType}
    if name in built_in_names:
        raise ValueError(f"'{name}' is a built-in constraint type and cannot be overridden.")
    _CUSTOM_CONSTRUCTOR_MAP[name] = cls
