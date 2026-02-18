"""Tests for DiscreteHMM and HMMFactory."""

import numpy as np

from linear_rnn_filtering.hmm import DiscreteHMM, HMMFactory


class TestDiscreteHMM:
    def test_random_init_stochastic(self):
        hmm = DiscreteHMM(3, 5)
        assert np.allclose(hmm.transfer_matrix.sum(axis=0), 1.0)
        assert np.allclose(hmm.emission_matrix.sum(axis=0), 1.0)

    def test_sample_shapes(self):
        hmm = HMMFactory.dishonest_casino()
        latent, emission = hmm.sample(batch_size=10, time_steps=50)
        assert latent.shape == (10, 50)
        assert emission.shape == (10, 50)

    def test_sample_values_in_range(self):
        hmm = HMMFactory.dishonest_casino()
        latent, emission = hmm.sample(batch_size=5, time_steps=100)
        assert np.all(latent >= 0) and np.all(latent < hmm.latent_dim)
        assert np.all(emission >= 0) and np.all(emission < hmm.emission_dim)

    def test_posterior_shapes(self):
        hmm = HMMFactory.dishonest_casino()
        _, emissions = hmm.sample(batch_size=4, time_steps=30)
        lp, ntp = hmm.compute_posterior(emissions)
        assert lp.shape == (4, 30, hmm.latent_dim)
        assert ntp.shape == (4, 30, hmm.emission_dim)

    def test_posterior_is_normalised(self):
        hmm = HMMFactory.dishonest_casino()
        _, emissions = hmm.sample(batch_size=2, time_steps=50)
        lp, ntp = hmm.compute_posterior(emissions)
        assert np.allclose(lp.sum(axis=-1), 1.0, atol=1e-6)
        assert np.allclose(ntp.sum(axis=-1), 1.0, atol=1e-6)

    def test_stationary_density_sums_to_one(self):
        hmm = HMMFactory.dishonest_casino()
        assert np.isclose(hmm.latent_stationary_density.sum(), 1.0)
        assert np.isclose(hmm.emission_stationary_density.sum(), 1.0)


class TestHMMFactory:
    def test_dishonest_casino_dims(self):
        hmm = HMMFactory.dishonest_casino()
        assert hmm.latent_dim == 2
        assert hmm.emission_dim == 6

    def test_random_dirichlet_dims(self):
        hmm = HMMFactory.random_dirichlet(latent_dim=4, emission_dim=8)
        assert hmm.latent_dim == 4
        assert hmm.emission_dim == 8

    def test_random_dirichlet_stochastic(self):
        hmm = HMMFactory.random_dirichlet(
            latent_dim=3, emission_dim=5, transfer_concentration=0.5, emission_concentration=0.5
        )
        assert np.allclose(hmm.transfer_matrix.sum(axis=0), 1.0)
        assert np.allclose(hmm.emission_matrix.sum(axis=0), 1.0)
