"""Training utilities for RNN sequence models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike

from .loss_functions import kl_divergence
from ..rnn import AbstractRNN

if TYPE_CHECKING:
    from ..hmm.models import AbstractHMM

__all__ = ["train"]

default_loss  = lambda rnn_outputs, rnn_latents, output_targets, latent_targets: kl_divergence(rnn_outputs, output_targets)
def train(
    rnn: AbstractRNN,
    get_batch: Callable[[], tuple[ArrayLike, ArrayLike | None, ArrayLike | None]],
    *,
    loss_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array] = default_loss,
    optimizer: float | Any = 0.01,
    num_epochs: int = 1,
    steps_per_epoch: int = 1,
    initial_condition: ArrayLike | None = None,
    callback: Callable[[int, float, AbstractRNN], None] | None = None,
) -> np.ndarray:
    """Train an RNN for ``num_epochs`` data-generation calls.

    Each epoch calls ``data_gen()`` once for fresh data, then takes
    ``steps_per_epoch`` gradient steps on that batch. Only unfrozen parameters
    are updated; call :meth:`AbstractRNN.freeze` before training to fix specific
    weights. The RNN is updated in-place.

    Args:
        rnn: The RNN to train. Updated in-place.
        get_batch: Callable returning ``(inputs, output_targets, latent_targets)``.
            Shapes: ``inputs`` ``(B, T, input_dim)``, targets ``(B, T, dim)`` or
            None. Return None for unused targets; ``loss_fn`` receives
            ``jnp.zeros(0)`` as a sentinel in that case.
        loss_fn: ``(rnn_output, rnn_latent, output_targets, latent_targets) -> scalar``.
            All arguments are ``(B, T, dim)`` arrays or ``jnp.zeros(0)``
            sentinels for unused targets.
        optimizer: Adam learning rate (float) or any ``optax.GradientTransformation``.
        num_epochs: Number of ``data_gen()`` calls (epochs).
        steps_per_epoch: Gradient steps taken on each batch. Defaults to 1.
            Values > 1 amortise the cost of data generation across multiple
            updates. All steps run on-device via ``jax.lax.scan``.
        initial_condition: Initial hidden state of shape ``(latent_dim,)``. Defaults to zeros.
        callback: Called as ``callback(epoch, loss, rnn)`` after each epoch,
            where ``loss`` is the mean over ``steps_per_epoch`` steps.

    Returns:
        np.ndarray: Loss history of shape ``(num_epochs,)``, one mean value per epoch.
    """
    if isinstance(optimizer, float):
        optimizer = optax.adam(optimizer)
    initial_condition = jnp.zeros(rnn.latent_dim) if initial_condition is None else jnp.asarray(initial_condition)

    forward = rnn.make_forward(initial_condition)
    trainable, static = rnn.partition()
    opt_state = optimizer.init(trainable)

    @jax.jit
    def update_epoch(diff, static, opt_state, inputs, output_targets, latent_targets):
        def compute_loss(d, s):
            rnn_output, rnn_latent = forward(d, s, inputs)
            return loss_fn(rnn_output, rnn_latent, output_targets, latent_targets)

        def one_step(carry, _):
            d, opt_s = carry
            loss_val, grads = jax.value_and_grad(compute_loss)(d, static)
            updates, new_opt_s = optimizer.update(grads, opt_s, d)
            return (optax.apply_updates(d, updates), new_opt_s), loss_val

        (new_diff, new_opt_state), step_losses = jax.lax.scan(
            one_step, (diff, opt_state), None, length=steps_per_epoch
        )
        return new_diff, new_opt_state, jnp.mean(step_losses)

    loss_history = np.zeros(num_epochs)
    current_diff = trainable

    for epoch in range(num_epochs):
        inputs, output_targets, latent_targets = get_batch()
        inputs = jnp.asarray(inputs)
        output_targets = jnp.zeros(0) if output_targets is None else jnp.asarray(output_targets)
        latent_targets = jnp.zeros(0) if latent_targets is None else jnp.asarray(latent_targets)

        current_diff, opt_state, loss_val = update_epoch(
            current_diff, static, opt_state, inputs, output_targets, latent_targets
        )
        rnn.combine(current_diff, static)
        loss_history[epoch] = float(loss_val)

        if callback is not None:
            callback(epoch, float(loss_val), rnn)

    return loss_history
