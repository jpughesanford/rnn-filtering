"""Tests for RNN models."""

import jax.numpy as jnp
import numpy as np
import pytest

from linear_rnn_filtering.rnn import AbstractRNN, ExactRNN, ModelA, ModelB
from linear_rnn_filtering.hmm import HMMFactory
from linear_rnn_filtering.types import LossType


@pytest.fixture
def casino():
    np.random.seed(42)
    return HMMFactory.dishonest_casino()


class TestConstruction:
    def test_exact_type(self):
        rnn = ExactRNN(2, 6)
        assert isinstance(rnn, ExactRNN)

    def test_model_a_type(self):
        rnn = ModelA(2, 6)
        assert isinstance(rnn, ModelA)

    def test_model_b_type(self):
        rnn = ModelB(2, 6)
        assert isinstance(rnn, ModelB)

    def test_model_a_raw_weight_names(self):
        rnn = ModelA(2, 6)
        assert "A_1" in rnn.raw_weight_names
        assert "A_2" in rnn.raw_weight_names
        assert "B" in rnn.raw_weight_names
        assert "C" in rnn.raw_weight_names

    def test_model_a_weight_names(self):
        rnn = ModelA(2, 6)
        assert rnn.weight_names == ["A", "B", "C"]

    def test_model_b_raw_weights(self):
        rnn = ModelB(2, 6)
        assert "d" in rnn.raw_weight_names

    def test_weights_property(self):
        rnn = ModelA(2, 6)
        w = rnn.weights
        assert "A" in w
        assert "B" in w
        assert "C" in w
        assert w["A"].shape == (2, 2)
        assert w["B"].shape == (2, 6)
        assert w["C"].shape == (6, 2)


class TestPredict:
    def test_predict_shapes(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        Y, X = rnn.predict(emissions)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)

    def test_predict_sums_to_one(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=2, time_steps=30)
        Y, _ = rnn.predict(emissions)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)

    def test_exact_initialized_matches_posterior(self, casino):
        rnn = ExactRNN(casino.latent_dim, casino.emission_dim)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=5, time_steps=200)
        _, ntp = casino.compute_posterior(emissions)
        x0 = np.log(casino.latent_stationary_density)
        Y, _ = rnn.predict(emissions, x0=x0)
        assert np.allclose(np.array(Y), ntp, atol=1e-6)

    def test_predict_with_x0(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=2, time_steps=20)
        x0 = np.ones(casino.latent_dim) * 0.5
        Y, X = rnn.predict(emissions, x0=x0)
        assert Y.shape == (2, 20, casino.emission_dim)


class TestInitializeFromHMM:
    def test_initialize_Astar(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_Astar(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=50)
        Y, _ = rnn.predict(emissions)
        assert Y.shape == (3, 50, casino.emission_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)

    def test_initialize_exact(self, casino):
        rnn = ExactRNN(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=50)
        Y, _ = rnn.predict(emissions)
        assert Y.shape == (3, 50, casino.emission_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)


class TestTraining:
    def test_train_on_posterior_with_kl_reduces_loss(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=1)
        loss = rnn.train(
            casino,
            loss=LossType.KL,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=50,
            print_every=999,
        )
        assert loss[-1, 0] < loss[0, 0]

    def test_train_on_posterior_with_hilbert_reduces_loss(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=1)
        loss = rnn.train(
            casino,
            loss=LossType.HILBERT,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=50,
            print_every=999,
        )
        assert loss[-1, 0] < loss[0, 0]

    def test_train_on_emissions_reduces_loss(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=2)
        loss = rnn.train(
            casino,
            loss=LossType.EMISSIONS,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=50,
            print_every=999,
        )
        assert loss[-1, 0] < loss[0, 0]


class TestFreezeUnfreeze:
    def test_freeze_prevents_update(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=3)
        rnn.freeze(["B", "C"])
        B_before = np.array(rnn.raw_weights["B"])
        C_before = np.array(rnn.raw_weights["C"])
        rnn.train(
            casino,
            batch_size=5,
            time_steps=20,
            optimization_steps=10,
            print_every=999,
        )
        assert np.allclose(B_before, np.array(rnn.raw_weights["B"]))
        assert np.allclose(C_before, np.array(rnn.raw_weights["C"]))
