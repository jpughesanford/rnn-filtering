"""A collection of Loss functions. Each loss function here should be paired with a :class:`LossType` to enforce strict
typing. This file exposes a `LOSS_MAP: dict[LossType,tuple[Callable,bool]]` object. LOSS_MAP[loss_type] returns both
the loss function of the corresponding type, as well as a boolean value indicating whether the loss function requires
computing the ground truth next token posterior distribution. Most functions require the posterior, but expected
surprisal does not."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from .parameters import Parameter
from .types import LossType

__all__ = ["LOSS_MAP"]

# ------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 5))
def expected_surprisal(
    batched_forward: Callable,
    parameters: dict[str, Parameter],
    emissions: jax.Array,
    true_next_token_posterior: jax.Array | None,
    x0: jax.Array,
    do_average: bool = True,
) -> jax.Array:
    """Mean negative log-likelihood of the next token.

    Computes ``-log p(y_{t+1} | y_{0:t})`` averaged over batch and time.

    Args:
        batched_forward (Callable): the RNN's class's batched forward function.
        parameters (dict[str, Parameter]): current parameters of the RNN.
        emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
        true_next_token_posterior (jax.Array): unused, but kept to keep loss function method signatures consistent.
        x0 (jax.Array): Initial hidden state of shape (latent_dim,).
        do_average (bool, optional): If True, return the scalar mean. If False,
            return per-timestep values. Defaults to True.

    Returns:
        jax.Array: Scalar mean NLL if ``do_average`` is True, otherwise array of shape (B, T-1).
    """
    _ = true_next_token_posterior
    output, _ = batched_forward(parameters, emissions, x0)
    output = output[:, :-1, :]
    next_token = emissions[:, 1:]
    probs = jnp.take_along_axis(output, next_token[..., None], axis=-1).squeeze(-1)
    nll = -jnp.log(probs + 1e-32)
    return jnp.mean(nll) if do_average else nll


@partial(jax.jit, static_argnums=(0, 5))
def expected_kl_divergence(
    batched_forward: Callable,
    parameters: dict[str, Parameter],
    emissions: jax.Array,
    true_next_token_posterior: jax.Array | None,
    x0: jax.Array,
    do_average: bool = True,
) -> jax.Array:
    """KL divergence from the ground-truth posterior to the RNN prediction.

    Computes ``KL(p_truth || p_rnn)`` at each time step.

    Args:
        batched_forward (Callable): the RNN's class's batched forward function.
        parameters (dict[str, Parameter]): current parameters of the RNN.
        emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
        true_next_token_posterior (jax.Array): Ground-truth posterior distributions of shape (B, T, emission_dim).
        x0 (jax.Array): Initial hidden state of shape (latent_dim,).
        do_average (bool, optional): If True, return the scalar mean. If False,
            return per-timestep values. Defaults to True.

    Returns:
        jax.Array: Scalar mean KL if ``do_average`` is True, otherwise array of shape (B, T).
    """
    p_rnn, _ = batched_forward(parameters, emissions, x0)
    eps = 1e-12
    p_rnn = p_rnn / jnp.sum(p_rnn, axis=-1, keepdims=True)
    p_truth = true_next_token_posterior / jnp.sum(true_next_token_posterior, axis=-1, keepdims=True)
    kl = jnp.sum(p_truth * (jnp.log(p_truth + eps) - jnp.log(p_rnn + eps)), axis=-1)
    return jnp.mean(kl) if do_average else kl


@partial(jax.jit, static_argnums=(0, 5))
def expected_hilbert_distance(
    batched_forward: Callable,
    parameters: dict[str, Parameter],
    emissions: jax.Array,
    true_next_token_posterior: jax.Array | None,
    x0: jax.Array,
    do_average: bool = True,
) -> jax.Array:
    """Hilbert projective metric between RNN output and target distributions.

    The Hilbert metric is defined as
    ``max_i log(p_rnn_i / p_truth_i) - min_i log(p_rnn_i / p_truth_i)``.

    Args:
        batched_forward (Callable): the RNN's class's batched forward function.
        parameters (dict[str, Parameter]): current parameters of the RNN.
        emissions (jax.Array): Emission indices of shape (B, T), dtype int32.
        true_next_token_posterior (jax.Array): Ground-truth posterior distributions of shape (B, T, emission_dim).
        x0 (jax.Array): Initial hidden state of shape (latent_dim,).
        do_average (bool, optional): If True, return the scalar mean. If False,
            return per-timestep values. Defaults to True.

    Returns:
        jax.Array: Scalar mean Hilbert distance if ``do_average`` is True, otherwise array of shape (B, T).
    """
    p_rnn, _ = batched_forward(parameters, emissions, x0)
    eps = 1e-16
    p_rnn = jnp.clip(p_rnn, eps, None)
    p_truth = jnp.clip(true_next_token_posterior, eps, None)
    log_ratio = jnp.log(p_rnn) - jnp.log(p_truth)
    hilbert = jnp.max(log_ratio, axis=-1) - jnp.min(log_ratio, axis=-1)
    return jnp.mean(hilbert) if do_average else hilbert


# ------------------------------------------------------------------
# Function Map to strictly assign each LossType to a loss function.
# ------------------------------------------------------------------

LOSS_MAP: dict[LossType, tuple[Callable, bool]] = {
    LossType.HILBERT: (expected_hilbert_distance, True),
    LossType.KL: (expected_kl_divergence, True),
    LossType.EMISSIONS: (expected_surprisal, False),
}
