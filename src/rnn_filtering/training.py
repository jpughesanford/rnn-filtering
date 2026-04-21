"""HMM-specific training utilities.

Provides :func:`train_on_hmm`, a batteries-included wrapper around
:meth:`AbstractRNN.train` that handles:

* sampling emission sequences from an :class:`~rnn_filtering.hmm.models.AbstractHMM`,
* embedding integer emission symbols as one-hot vectors (the RNN input),
* computing ground-truth posteriors when required by the chosen loss, and
* looping over multiple data-resampling epochs.

The RNN class itself is HMM-agnostic; all HMM coupling lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .rnn.loss_functions import LOSS_MAP
from .rnn.models import AbstractRNN
from .rnn.types import LossType

if TYPE_CHECKING:
    from .hmm.models import AbstractHMM

__all__ = ["train_on_hmm"]


def train_on_hmm(
    rnn: AbstractRNN,
    hmm: AbstractHMM,
    *,
    output_loss: str | Callable = LossType.KL,
    latent_loss: str | Callable | None = None,
    batch_size: int = 100,
    time_steps: int = 1000,
    num_epochs: int = 1,
    learning_rate: float = 2e-2,
    optimization_steps: int = 2000,
    print_every: int = 100,
    x0: ArrayLike | None = None,
) -> np.ndarray:
    """Train an RNN on data sampled from an :class:`~rnn_filtering.hmm.models.AbstractHMM`.

    Emission symbols are embedded as one-hot vectors before being fed to the
    RNN.  When ``output_loss`` is ``'emissions'``, the same one-hot embeddings
    are used as the training targets (equivalent to minimising NLL).  For all
    other string loss types the HMM posterior is computed and used as the
    target.

    Args:
        rnn (AbstractRNN): The RNN to train (modified in-place).
        hmm (AbstractHMM): HMM used to generate training data.
        output_loss (str | Callable): Loss applied to the RNN output.  Pass
            ``'emissions'`` to train against one-hot emission targets (NLL).
            Pass ``'kl'`` (default) or ``'hilbert'`` to train against the HMM
            posterior.  A custom callable ``(result, desired) -> scalar`` is
            also accepted; in that case the HMM posterior is computed and passed
            as ``desired_output``.
        latent_loss (str | Callable | None): Optional loss on the RNN latent
            state.  Same callable contract as ``output_loss``.  Defaults to
            None.
        batch_size (int): Number of independent trajectories per epoch.
            Defaults to 100.
        time_steps (int): Length of each sampled trajectory. Defaults to 1000.
        num_epochs (int): Number of data-resampling epochs. Defaults to 1.
        learning_rate (float): Adam learning rate. Defaults to 2e-2.
        optimization_steps (int): Gradient steps per epoch. Defaults to 2000.
        print_every (int): Print loss every this many steps (0 = silent).
            Defaults to 100.
        x0 (ArrayLike, optional): Initial hidden state of shape (latent_dim,).
            Defaults to zeros.

    Returns:
        np.ndarray: Loss history of shape (optimization_steps, num_epochs).
    """

    loss_history = np.zeros((optimization_steps, num_epochs))
    train_on_emissions = isinstance(output_loss, str) and LossType(output_loss) == LossType.EMISSIONS
    resolved_output_loss: str | Callable = LossType.KL if train_on_emissions else output_loss

    for epoch in range(num_epochs):
        if num_epochs > 1:
            print(f"epoch {epoch + 1}/{num_epochs}")

        # Sample and embed.
        _, emissions = hmm.sample(batch_size, time_steps)
        emissions = jnp.asarray(emissions, dtype=jnp.int32)
        inputs = jax.nn.one_hot(emissions, rnn.input_dim)  # (B, T, K)

        # Build one hot output.
        if train_on_emissions:
            # KL(one_hot(e) ‖ q) = -log q[e] = NLL; no posterior needed.
            desired_output: jax.Array | None = inputs
        elif callable(output_loss) or (isinstance(output_loss, str) and output_loss in LOSS_MAP):
            # Posterior-based loss (kl, hilbert, one_norm, or custom callable).
            _, desired_output = hmm.compute_posterior(emissions)
        else:
            desired_output = None

        epoch_hist = rnn.train(
            inputs,
            desired_output=desired_output,
            latent_loss=latent_loss,
            output_loss=resolved_output_loss,
            learning_rate=learning_rate,
            optimization_steps=optimization_steps,
            print_every=print_every,
            x0=x0,
        )

        loss_history[:, epoch] = epoch_hist

    return loss_history
