"""Loss functions for RNN training and evaluation.

Each function has the signature::

    fn(p, q, *, clip=1e-12, do_average=True) -> jax.Array

where value1 and value2 are of shape ``(Batch, Timesteps, Dimension)``. Custom loss callables must match this
signature. When do_average=True, the method must return a scalar.
"""

import jax.numpy as jnp

__all__ = ["kl_divergence", "hilbert_distance", "one_norm"]


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------


def kl_divergence(
    p: jnp.ndarray,
    q: jnp.ndarray,
    *,
    clip: float = 1e-12,
    do_average: bool = True,
) -> jnp.ndarray:
    """KL divergence KL(p ‖ q).

    Uses the convention ``0 * log(0) = 0``. Anything less than clip is considered zero.
    One-hot vectors are valid; KL(one_hot(e) ‖ q) = -log q[e].

    Args:
        p (jnp.ndarray): shape (B, T, D). must be non-negative and stochastic over D.
        q (jnp.ndarray): shape (B, T, D). must be non-negative and stochastic over D.

        clip (float): Small constant. Any p value beneath this is considered zero. Defaults to 1e-12.
        do_average (bool): If True, return the scalar mean over B and T.
            If False, return per-timestep values of shape (B, T). Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean KL if ``do_average``, else shape (B, T).
    """
    p = jnp.where(p < clip, 0.0, p)
    value = jnp.where(p > 0, p * (jnp.log(jnp.maximum(p, clip)) - jnp.log(jnp.maximum(q, clip))), 0.0).sum(axis=-1)
    return jnp.mean(value) if do_average else value


def hilbert_distance(
    p: jnp.ndarray,
    q: jnp.ndarray,
    *,
    clip: float = 1e-12,
    do_average: bool = True,
) -> jnp.ndarray:
    """Hilbert projective metric between ``result`` and ``desired``.

    Defined as ``max_k log(result_k / desired_k) - min_k log(result_k / desired_k)``.

    Args:
        p (jnp.ndarray): shape (B, T, D). must be non-negative.
        q (jnp.ndarray): shape (B, T, D). must be non-negative.
        clip (float): Small constant. Any value beneath this is considered zero. Defaults to 1e-12.
        do_average (bool): If True, return the scalar mean. Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean Hilbert distance if ``do_average``, else shape (B, T).
    """
    p = jnp.where(p < clip, 0.0, p)
    q = jnp.where(q < clip, 0.0, q)
    log_ratio = jnp.where(p + q > 0, jnp.log(jnp.maximum(p, clip)) - jnp.log(jnp.maximum(q, clip)), 0.0)
    value = jnp.max(log_ratio, axis=-1) - jnp.min(log_ratio, axis=-1)
    return jnp.mean(value) if do_average else value


def one_norm(
    p: jnp.ndarray,
    q: jnp.ndarray,
    *,
    do_average: bool = True,
) -> jnp.ndarray:
    """L1 norm between ``result`` and ``desired`` distributions.

    Both arrays are L1-normalised before comparison.

    Args:
        p (jnp.ndarray): shape (B, T, D). 
        q (jnp.ndarray): shape (B, T, D). 
        do_average (bool): If True, return the scalar mean. Defaults to True.

    Returns:
        jnp.ndarray: Scalar mean L1 distance if ``do_average``, else shape (B, T).
    """
    value = jnp.sum(jnp.abs(p - q), axis=-1)
    return jnp.mean(value) if do_average else value