"""Tests for RNN models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rnn_filtering.hmm import HMMFactory
from rnn_filtering.rnn import AbstractRNN, ExactRNN, ModelA, ModelB, Parameter
from rnn_filtering.rnn.parameters import NonnegativeParameter, StableParameter, StochasticParameter
from rnn_filtering.rnn.types import ConstraintType, LossType
from rnn_filtering.training import train_on_hmm


@pytest.fixture
def casino():
    return HMMFactory.dishonest_casino()


class TestConstruction:
    def test_schema(self):
        class TestRNN(AbstractRNN):
            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "stable"},
                    "B": {"shape": (latent_dim, input_dim), "constraint": ConstraintType.STOCHASTIC},
                    "C": {"constraint": "nonnegative"},
                    "D": {"shape": (input_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = C @ x_t
                return x_t, y_t

        test_rnn = TestRNN(10, 10, 10)
        assert isinstance(test_rnn._parameters["A"], StableParameter)
        assert isinstance(test_rnn._parameters["B"], StochasticParameter)
        assert isinstance(test_rnn._parameters["C"], NonnegativeParameter)
        assert isinstance(test_rnn._parameters["D"], Parameter)
        assert test_rnn._parameters["C"].shape == (1,)

    def test_exact_initializes_with_correct_architecture(self):
        latent_dim = 2
        input_dim = 2
        output_dim = 2
        rnn = ExactRNN(input_dim, latent_dim, output_dim)
        assert isinstance(rnn, ExactRNN)
        assert rnn.get_parameter_names() == {"A", "B", "C"}
        assert isinstance(rnn._parameters["A"], StochasticParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], StochasticParameter)
        values = rnn.get_parameter_values({"A", "B", "C"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, input_dim)
        assert values["C"].shape == (output_dim, latent_dim)

    def test_model_a_initializes_with_correct_architecture(self):
        latent_dim = 5
        input_dim = 6
        output_dim = 6
        rnn = ModelA(input_dim, latent_dim, output_dim)
        assert isinstance(rnn, ModelA)
        assert rnn.get_parameter_names() == {"A", "B", "C"}
        assert isinstance(rnn._parameters["A"], StableParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], StochasticParameter)
        values = rnn.get_parameter_values({"A", "B", "C"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, input_dim)
        assert values["C"].shape == (output_dim, latent_dim)

    def test_model_b_initializes_with_correct_architecture(self):
        latent_dim = 2
        input_dim = 2
        output_dim = 2
        rnn = ModelB(input_dim, latent_dim, output_dim)
        assert isinstance(rnn, ModelB)
        assert rnn.get_parameter_names() == {"A", "B", "C", "d"}
        assert isinstance(rnn._parameters["A"], StableParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], Parameter)
        assert isinstance(rnn._parameters["d"], Parameter)
        values = rnn.get_parameter_values({"A", "B", "C", "d"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, input_dim)
        assert values["C"].shape == (output_dim, latent_dim)
        assert values["d"].shape == (output_dim,)


class TestPredict:
    def test_respond(self, casino):
        rnn = ExactRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        _, posterior = casino.compute_posterior(emissions)
        x0 = np.log(casino.latent_stationary_density)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, X = rnn.respond(inputs, x0=x0)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)
        assert np.allclose(np.array(Y), posterior, atol=1e-6)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)

    def test_model_b_respond(self, casino):
        rnn = ModelB(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, X = rnn.respond(inputs)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)


class TestInitializeFromHMM:
    def test_initialize_astar(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_astar(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=50)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, _ = rnn.respond(inputs)
        assert Y.shape == (3, 50, casino.emission_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)

    def test_initialize_exact(self, casino):
        rnn = ExactRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=50)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, _ = rnn.respond(inputs)
        assert Y.shape == (3, 50, casino.emission_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)


class TestSampleLoss:
    def test_shapes_and_signs(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=5, time_steps=20)
        emissions = jnp.asarray(emissions, jnp.int32)
        inputs = jax.nn.one_hot(emissions, casino.emission_dim)
        _, desired = casino.compute_posterior(emissions)

        kl = rnn.sample_loss(inputs, desired_output=desired, output_loss=LossType.KL)
        hilbert = rnn.sample_loss(inputs, desired_output=desired, output_loss=LossType.HILBERT)
        one_hot_kl = rnn.sample_loss(inputs, desired_output=inputs, output_loss=LossType.KL)

        assert kl.shape == (5, 20)
        assert hilbert.shape == (5, 20)
        assert one_hot_kl.shape == (5, 20)
        assert jnp.all(kl >= 0)
        assert jnp.all(hilbert >= 0)
        assert jnp.all(one_hot_kl >= 0)


class TestTraining:
    def test_train_on_posterior_with_kl_reduces_loss(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=1)
        loss = train_on_hmm(
            rnn,
            casino,
            output_loss=LossType.KL,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=50,
            print_every=0,
        )
        assert loss[-1, 0] < 0.9 * loss[0, 0]

    def test_train_on_posterior_with_hilbert_reduces_loss(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=1)
        loss = train_on_hmm(
            rnn,
            casino,
            output_loss=LossType.HILBERT,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=50,
            print_every=0,
        )
        assert loss[-1, 0] < 0.9 * loss[0, 0]

    def test_train_on_emissions_reduces_loss(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=2)
        loss = train_on_hmm(
            rnn,
            casino,
            output_loss=LossType.EMISSIONS,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=1000,
            print_every=0,
        )
        assert loss[-1, 0] < loss[0, 0]

    def test_train_returns_correct_shape(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        loss = train_on_hmm(rnn, casino, num_epochs=2, optimization_steps=10, print_every=0)
        assert loss.shape == (10, 2)

    def test_train_directly_with_precomputed_data(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=1)
        _, emissions = casino.sample(10, 50)
        emissions = jnp.asarray(emissions, jnp.int32)
        inputs = jax.nn.one_hot(emissions, casino.emission_dim)
        _, desired = casino.compute_posterior(emissions)
        loss = rnn.train(
            inputs,
            desired_output=desired,
            output_loss="kl",
            optimization_steps=20,
            print_every=0,
        )
        assert loss.shape == (20,)
        assert loss[-1] < loss[0]


class TestFreezeUnfreeze:
    def test_freeze_prevents_update(self, casino):
        rnn = ModelA(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=3)
        all_names = rnn.get_parameter_names()
        frozen = {"B", "C"}
        unfrozen = all_names - frozen
        rnn.freeze(frozen)
        before = rnn.get_parameter_values(all_names)
        train_on_hmm(
            rnn,
            casino,
            batch_size=5,
            time_steps=20,
            optimization_steps=10,
            print_every=0,
        )
        after = rnn.get_parameter_values(all_names)
        for name in frozen:
            assert np.allclose(before[name], after[name])
        for name in unfrozen:
            assert not np.allclose(before[name], after[name])
