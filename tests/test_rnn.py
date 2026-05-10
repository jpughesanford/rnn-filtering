"""Tests for RNN models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rnn_filtering.hmm import HMMFactory
from rnn_filtering.rnn import AbstractRNN, ExactRNN, LinearRNN, Parameter
from rnn_filtering.rnn.parameters import NonnegativeParameter, StableParameter, StochasticParameter
from rnn_filtering.rnn.types import ConstraintType
from rnn_filtering.train.loss_functions import kl_divergence
from rnn_filtering.train.utils import train


@pytest.fixture
def casino():
    return HMMFactory.dishonest_casino()


def _get_batch(casino, rnn, batch_size=10, time_steps=50):
    def get_batch():
        _, emissions = casino.sample(batch_size, time_steps)
        emissions = jnp.asarray(emissions, jnp.int32)
        inputs = jax.nn.one_hot(emissions, rnn.input_dim)
        _, posterior = casino.compute_posterior(emissions)
        return inputs, jnp.asarray(posterior), None
    return get_batch


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

    def test_linear_rnn_initializes_with_correct_architecture(self):
        latent_dim = 5
        input_dim = 6
        output_dim = 6
        rnn = LinearRNN(input_dim, latent_dim, output_dim)
        assert isinstance(rnn, LinearRNN)
        assert rnn.get_parameter_names() == {"A", "B", "C"}
        assert isinstance(rnn._parameters["A"], StableParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], StochasticParameter)
        values = rnn.get_parameter_values({"A", "B", "C"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, input_dim)
        assert values["C"].shape == (output_dim, latent_dim)


class TestPredict:
    def test_respond(self, casino):
        rnn = ExactRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        _, posterior = casino.compute_posterior(emissions)
        x0 = np.log(casino.latent_stationary_density)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, X = rnn.respond(inputs, initial_condition=x0)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)
        assert np.allclose(np.array(Y), posterior, atol=1e-6)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)


class TestInitializeFromHMM:
    def test_initialize_astar(self, casino):
        rnn = LinearRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
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


class TestTraining:
    def test_train_reduces_kl_loss(self, casino):
        rnn = LinearRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=1)
        loss = train(rnn, _get_batch(casino, rnn), num_epochs=5, steps_per_epoch=50, optimizer=1e-2)
        assert loss[-1] < 0.9 * loss[0]

    def test_train_returns_correct_shape(self, casino):
        rnn = LinearRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        loss = train(rnn, _get_batch(casino, rnn), num_epochs=3, steps_per_epoch=5)
        assert loss.shape == (3,)

    def test_train_with_custom_loss(self, casino):
        rnn = LinearRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=1)
        loss_fn = lambda out, lat, tgt_out, tgt_lat: kl_divergence(out, tgt_out)
        loss = train(rnn, _get_batch(casino, rnn), loss_fn=loss_fn, num_epochs=5, steps_per_epoch=50, optimizer=1e-2)
        assert loss[-1] < 0.9 * loss[0]

    def test_exact_rnn_converges_to_near_zero_kl(self, casino):
        rnn = ExactRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        x0 = jnp.log(jnp.asarray(casino.latent_stationary_density))
        loss = train(
            rnn,
            _get_batch(casino, rnn, batch_size=20, time_steps=100),
            num_epochs=5,
            steps_per_epoch=200,
            optimizer=1e-2,
            initial_condition=x0,
        )
        assert loss[-1] < 0.05


class TestFreezeUnfreeze:
    def test_freeze_prevents_update(self, casino):
        rnn = LinearRNN(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=3)
        all_names = rnn.get_parameter_names()
        frozen = {"B", "C"}
        unfrozen = all_names - frozen
        rnn.freeze(frozen)
        before = rnn.get_parameter_values(all_names)
        train(rnn, _get_batch(casino, rnn, batch_size=5, time_steps=20), num_epochs=1, steps_per_epoch=10)
        after = rnn.get_parameter_values(all_names)
        for name in frozen:
            assert np.allclose(before[name], after[name])
        for name in unfrozen:
            assert not np.allclose(before[name], after[name])