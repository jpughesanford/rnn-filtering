"""RNN architectures for sequence modelling.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import logsumexp

from .abstract import AbstractRNN
from .types import ConstraintType, Schema

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rnn_filtering.hmm.models import AbstractHMM as GenericHMM

__all__ = [ "ExactRNN", "LinearRNN"]


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
# Linear RNN  (stable linear recurrence, stochastic linear of softmax readout. See paper for more. )
# ---------------------------------------------------------------------------


class LinearRNN(AbstractRNN):
    """Linear RNN with stable latent dynamics and stochastic readout.

    Recurrence: ``x_t = A x_{t-1} + B input_t``;
    readout: ``p_t = C softmax(x_t)``.

    ``A`` is parameterised via the Cayley transform to guarantee spectral
    radius <= 1.  ``C`` is softmax-normalised to a column-stochastic matrix.
    """

    @staticmethod
    def schema(input_dim: int, latent_dim: int, output_dim: int) -> Schema:
        if input_dim != output_dim:
            raise ValueError(f"LinearRNN requires input_dim == output_dim, got {input_dim} != {output_dim}.")
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

