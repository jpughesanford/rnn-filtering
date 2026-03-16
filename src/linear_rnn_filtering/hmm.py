"""Discrete Hidden Markov Model, implemented in JAX, with batch sampling and exact forward filtering.
Also includes HMMFactory methods for constructing standard HMM instances"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["DiscreteHMM", "HMMFactory"]


class DiscreteHMM:
    """A discrete time, discrete-state Hidden Markov Model, with node-determined emissions.

    Matrices use column-stochastic convention: columns sum to one.
    Internally all matrices and distributions are stored as JAX arrays.

    Attributes:
        transfer_matrix (jax.Array): Transition matrix of shape (latent_dim, latent_dim).
        emission_matrix (jax.Array): Emission matrix of shape (emission_dim, latent_dim).
        latent_stationary_density (jax.Array): Stationary distribution of the latent Markov chain,
            shape (latent_dim,).
        emission_stationary_density (jax.Array): Marginal emission distribution under stationarity,
            shape (emission_dim,).
    """

    def __init__(self, latent_dim: int, emission_dim: int) -> None:
        """Initialize a discrete HMM. Transfer and emission matrices are initialized randomly,
        and should be set with :meth:`set_transfer_matrix` and :meth:`set_emission_matrix`, respectively.

        Args:
            latent_dim (int): Number of hidden states.
            emission_dim (int): Number of emission states.
        """
        self.latent_dim: int = latent_dim
        self.emission_dim: int = emission_dim

        self.transfer_matrix: jax.Array | None = None
        self.emission_matrix: jax.Array | None = None
        self.latent_stationary_density: jax.Array | None = None
        self.emission_stationary_density: jax.Array | None = None

        transfer_matrix = np.random.rand(latent_dim, latent_dim)
        transfer_matrix = transfer_matrix / transfer_matrix.sum(axis=0, keepdims=True)
        self.set_transfer_matrix(transfer_matrix)

        emission_matrix = np.random.rand(emission_dim, latent_dim)
        emission_matrix = emission_matrix / emission_matrix.sum(axis=0, keepdims=True)
        self.set_emission_matrix(emission_matrix)

    def sample(
        self,
        batch_size: int = 1,
        time_steps: int = 100,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample hidden-state and emission trajectories.

        Args:
            batch_size (int, optional): Number of independent timeseries. Defaults to 1.
            time_steps (int, optional): Length of each timeseries. Defaults to 100.
            key (jax.Array, optional): JAX PRNG key. If None, a new key is randomly generated.

        Returns:
            latent (jax.Array): Hidden state indices, shape (batch_size, time_steps).
                Values range from 0 to latent_dim-1.
            emission (jax.Array): Observed emission indices, shape (batch_size, time_steps).
                Values range from 0 to emission_dim-1.
        """
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        return _sample_scan(
            self.transfer_matrix,
            self.emission_matrix,
            self.latent_stationary_density,
            batch_size,
            time_steps,
            key,
        )

    def compute_posterior(self, emissions: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Compute exact forward-filtered posteriors.

        Args:
            emissions (ArrayLike): Observed emission indices, shape (batch_size, time_steps).

        Returns:
            latent_posterior (jax.Array): Posterior over hidden states, shape (batch_size, time_steps, latent_dim).
                Represents P(x_t | y_{0:t}) for each time step.
            next_emission_posterior (jax.Array): Posterior over next emissions,
                shape (batch_size, time_steps, emission_dim).
                Represents P(y_{t+1} | y_{0:t}) for each time step.
        """

        # Runs the forward algorithm in log-space for numerical stability. Output is in probability-space.
        init_state = jnp.log(self.latent_stationary_density)
        em = jnp.asarray(emissions, dtype=jnp.int32)

        latent_posterior, next_emission_posterior = _forward_filter_scan(
            self.transfer_matrix, self.emission_matrix, init_state, em
        )
        return latent_posterior, next_emission_posterior

    def set_transfer_matrix(self, transfer_matrix: ArrayLike) -> None:
        """Set the latent-to-latent transition matrix.

        Validates column-stochasticity, computes the stationary
        distribution via eigendecomposition, and stores both as JAX arrays.

        Args:
            transfer_matrix (ArrayLike): Column-stochastic transition matrix of shape (latent_dim, latent_dim).

        Raises:
            ValueError: If the matrix has the wrong shape, is not column-stochastic,
                or the stationary eigenvector is not real-valued.
        """
        transfer_matrix = np.asarray(transfer_matrix)
        if transfer_matrix.shape != (self.latent_dim, self.latent_dim):
            raise ValueError(f"Transfer matrix has wrong shape. Expected {(self.latent_dim, self.latent_dim)}, got {transfer_matrix.shape}.")
        colsum = transfer_matrix.sum(axis=0)
        if not np.allclose(colsum, 1, 1e-16):
            raise ValueError(f"Transfer matrix is not stochastic. Worst column sum deviation from one: {colsum.max()}")
        eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)
        idx = np.argmin(np.abs(eigenvalues - 1))
        if np.linalg.norm(np.imag(eigenvectors[:, idx])) > 1e-8:
            raise ValueError(f"Leading eigenvector appears to be imaginary")
        assert np.linalg.norm(np.imag(eigenvectors[:, idx])) < 1e-8
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        self.transfer_matrix = jnp.asarray(transfer_matrix)
        self.latent_stationary_density = jnp.asarray(stationary)

    def set_emission_matrix(self, emission_matrix: ArrayLike) -> None:
        """Set the latent-to-emission matrix.

        Validates column-stochasticity and stores the matrix as a JAX
        array. Also recomputes the marginal emission distribution.

        Args:
            emission_matrix (ArrayLike): Column-stochastic emission matrix of shape (emission_dim, latent_dim).

        Raises:
            ValueError: If the matrix has the wrong shape or is not column-stochastic.
        """
        emission_matrix = np.asarray(emission_matrix)
        assert emission_matrix.shape == (self.emission_dim, self.latent_dim)
        assert np.allclose(emission_matrix.sum(axis=0), 1, 1e-16)
        self.emission_matrix = jnp.asarray(emission_matrix)
        self.emission_stationary_density = self.emission_matrix @ self.latent_stationary_density


@partial(jax.jit, static_argnums=(3, 4))
def _sample_scan(
    transfer_matrix: jax.Array,
    emission_matrix: jax.Array,
    pi: jax.Array,
    batch_size: int,
    time_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled HMM sampling via `jax.lax.scan` and `jax.vmap`.

    Args:
        transfer_matrix (jax.Array): Column-stochastic transfer matrix of shape (n, n).
        emission_matrix (jax.Array): Column-stochastic emission matrix of shape (m, n).
        pi (jax.Array): Stationary distribution over latent states of shape (n,).
        batch_size (int): Number of independent trajectories.
        time_steps (int): Length of each trajectory.
        key (jax.Array): JAX PRNG key.

    Returns:
        latent (jax.Array): Sampled hidden state indices, shape (batch_size, time_steps), dtype int32.
        emissions (jax.Array): Sampled emission indices, shape (batch_size, time_steps), dtype int32.
    """
    log_transfer_matrix = jnp.log(transfer_matrix)
    log_emission_matrix = jnp.log(emission_matrix)
    log_pi = jnp.log(pi)

    def sample_single(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key, k0, k1 = jax.random.split(key, 3)
        init_state = jax.random.categorical(k0, log_pi)
        init_emission = jax.random.categorical(k1, log_emission_matrix[:, init_state])

        def step(carry, _):
            state, key = carry
            key, k_state, k_emit = jax.random.split(key, 3)
            next_state = jax.random.categorical(k_state, log_transfer_matrix[:, state])
            emission = jax.random.categorical(k_emit, log_emission_matrix[:, next_state])
            return (next_state, key), (next_state, emission)

        (_, _), (states, emissions) = jax.lax.scan(step, (init_state, key), None, length=time_steps - 1)
        states = jnp.concatenate([init_state[None], states])
        emissions = jnp.concatenate([init_emission[None], emissions])
        return states, emissions

    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample_single)(keys)


@jax.jit
def _forward_filter_scan(
    transfer_matrix: jax.Array,
    emission_matrix: jax.Array,
    init_state: jax.Array,
    emissions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled forward filtering using `jax.lax.scan` over time and `jax.vmap` over batch.

    Args:
        transfer_matrix (jax.Array): Column-stochastic transfer matrix of shape (n, n).
        emission_matrix (jax.Array): Column-stochastic emission matrix of shape (m, n).
        init_state (jax.Array): Log stationary distribution (may contain -inf), shape (n,).
        emissions (jax.Array): Observed emission indices, shape (batch, timesteps).

    Returns:
        latent_posterior (jax.Array): Filtered latent-state posteriors, shape (batch, timesteps, n).
        next_emission_posterior (jax.Array): One-step-ahead emission predictive distributions,
            shape (batch, timesteps, m).
    """
    log_transfer_matrix = jnp.log(transfer_matrix)
    log_emission_matrix = jnp.log(emission_matrix)

    def step(state: jax.Array, emission_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        pred = jax.nn.logsumexp(log_transfer_matrix + state[None, :], axis=1)
        state = pred + log_emission_matrix[emission_t, :]
        state = state - jax.nn.logsumexp(state)
        return state, state

    def scan_single(emissions_single: jax.Array) -> jax.Array:
        _, log_posteriors = jax.lax.scan(step, init_state, emissions_single)
        return log_posteriors

    all_log_post = jax.vmap(scan_single)(emissions)  # (batch, timesteps, n)
    latent_posterior = jnp.exp(all_log_post)

    next_token_posterior = jnp.einsum("ij,btj->bti", transfer_matrix, latent_posterior)
    next_token_posterior = jnp.einsum("ij,btj->bti", emission_matrix, next_token_posterior)

    return latent_posterior, next_token_posterior


class HMMFactory:
    """Factory for creating pre-configured `DiscreteHMM` instance types."""

    @staticmethod
    def dishonest_casino() -> DiscreteHMM:
        """Construct the "dishonest casino" HMM.

        This is a two-state HMM modeling a fair die versus a loaded die
        (biased toward 6). The transition matrix has high self-transition
        probabilities, representing sticky latent states.

        Returns:
            DiscreteHMM: HMM instance
        """
        hmm = DiscreteHMM(2, 6)
        hmm.set_transfer_matrix(
            np.array(
                [
                    [0.95, 0.10],
                    [0.05, 0.90],
                ]
            )
        )
        hmm.set_emission_matrix(
            np.array(
                [
                    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                    [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
                ]
            ).T
        )
        return hmm

    @staticmethod
    def random_dirichlet(
        latent_dim: int = 2,
        emission_dim: int = 2,
        transfer_concentration: float = 0.9,
        emission_concentration: float = 0.9,
    ) -> DiscreteHMM:
        """Construct a random HMM with Dirichlet-sampled matrices.

        The latent-to-latent transition matrix and latent-to-emission
        matrix are constructed from columns sampled independently from
        Dirichlet distributions with concentrations `transfer_concentration`
        and `emission_concentration`, respectively.

        Args:
            latent_dim (int, optional): Number of hidden states. Defaults to 2.
            emission_dim (int, optional): Number of emission states. Defaults to 2.
            transfer_concentration (float, optional): Dirichlet concentration parameter for the transition matrix.
                Values < 1 produce peaky more sparse matrices; values > 1 produce more uniform matrices
            emission_concentration (float, optional): Dirichlet concentration parameter for the emission matrix.
                Values < 1 produce peaky more sparse matrices; values > 1 produce more uniform matrices

        Returns:
            DiscreteHMM: HMM instance
        """
        hmm = DiscreteHMM(latent_dim=latent_dim, emission_dim=emission_dim)
        transfer_matrix = np.random.dirichlet(transfer_concentration * np.ones(latent_dim), size=latent_dim).T
        hmm.set_transfer_matrix(transfer_matrix)
        emission_matrix = np.random.dirichlet(emission_concentration * np.ones(emission_dim), size=latent_dim).T
        hmm.set_emission_matrix(emission_matrix)
        return hmm
