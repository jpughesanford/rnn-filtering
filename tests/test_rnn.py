"""Tests for RNN models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_rnn_filtering.hmm import HMMFactory
from linear_rnn_filtering.parameters import (
    NonnegativeParameter,
    Parameter,
    StableParameter,
    StochasticParameter,
    register_parameter_type,
)
from linear_rnn_filtering.rnn import AbstractRNN, ExactRNN, ModelA, ModelB
from linear_rnn_filtering.types import ConstraintType, LossType


@pytest.fixture
def casino():
    return HMMFactory.dishonest_casino()


class TestConstruction:
    def test_schema(self):
        class TestRNN(AbstractRNN):
            @staticmethod
            def schema(n, m):
                return {
                    "A": {"shape": (n, n), "constraint": "stable"},
                    "B": {"shape": (n, m), "constraint": ConstraintType.STOCHASTIC},
                    "C": {"constraint": "nonnegative"},
                    "D": {"shape": (m, n)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = C @ x_t
                return x_t, y_t

        test_rnn = TestRNN(10, 10)
        assert isinstance(test_rnn._parameters["A"], StableParameter)
        assert isinstance(test_rnn._parameters["B"], StochasticParameter)
        assert isinstance(test_rnn._parameters["C"], NonnegativeParameter)
        assert isinstance(test_rnn._parameters["D"], Parameter)
        assert test_rnn._parameters["C"].shape == (1,)

    def test_exact_initializes_with_correct_architecture(self):
        latent_dim = 2
        emission_dim = 2
        rnn = ExactRNN(latent_dim, emission_dim)
        assert isinstance(rnn, ExactRNN)
        assert rnn.get_parameter_names() == {"A", "B", "C"}
        assert isinstance(rnn._parameters["A"], StochasticParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], StochasticParameter)
        values = rnn.get_parameter_values({"A", "B", "C"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, emission_dim)
        assert values["C"].shape == (emission_dim, latent_dim)

    def test_model_a_initializes_with_correct_architecture(self):
        latent_dim = 5
        emission_dim = 6
        rnn = ModelA(latent_dim, emission_dim)
        assert isinstance(rnn, ModelA)
        assert rnn.get_parameter_names() == {"A", "B", "C"}
        assert isinstance(rnn._parameters["A"], StableParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], StochasticParameter)
        values = rnn.get_parameter_values({"A", "B", "C"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, emission_dim)
        assert values["C"].shape == (emission_dim, latent_dim)

    def test_model_b_initializes_with_correct_architecture(self):
        latent_dim = 2
        emission_dim = 2
        rnn = ModelB(latent_dim, emission_dim)
        assert isinstance(rnn, ModelB)
        assert rnn.get_parameter_names() == {"A", "B", "C", "d"}
        assert isinstance(rnn._parameters["A"], StableParameter)
        assert isinstance(rnn._parameters["B"], Parameter)
        assert isinstance(rnn._parameters["C"], Parameter)
        assert isinstance(rnn._parameters["d"], Parameter)
        values = rnn.get_parameter_values({"A", "B", "C", "d"})
        assert values["A"].shape == (latent_dim, latent_dim)
        assert values["B"].shape == (latent_dim, emission_dim)
        assert values["C"].shape == (emission_dim, latent_dim)
        assert values["d"].shape == (emission_dim,)


class TestPredict:
    def test_predict(self, casino):
        rnn = ExactRNN(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_weights(casino)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        _, posterior = casino.compute_posterior(emissions)
        x0 = np.log(casino.latent_stationary_density)
        Y, X = rnn.predict(emissions, x0=x0)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)
        assert np.allclose(np.array(Y), posterior, atol=1e-6)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)

    def test_model_b_predict(self, casino):
        rnn = ModelB(casino.latent_dim, casino.emission_dim, seed=0)
        _, emissions = casino.sample(batch_size=3, time_steps=20)
        Y, X = rnn.predict(emissions)
        assert Y.shape == (3, 20, casino.emission_dim)
        assert X.shape == (3, 20, casino.latent_dim)
        assert np.allclose(jnp.sum(Y, axis=-1), 1.0, atol=1e-5)


class TestInitializeFromHMM:
    def test_initialize_astar(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.initialize_astar(casino)
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


class TestSampleLoss:
    def test_shapes_and_signs(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=0)
        kl = rnn.sample_loss(casino, loss=LossType.KL, batch_size=5, time_steps=20)
        hilbert = rnn.sample_loss(casino, loss=LossType.HILBERT, batch_size=5, time_steps=20)
        emissions = rnn.sample_loss(casino, loss=LossType.EMISSIONS, batch_size=5, time_steps=20)
        assert kl.shape == (5, 20)
        assert hilbert.shape == (5, 20)
        assert emissions.shape == (5, 19)  # T-1: last timestep has no next-token target
        assert jnp.all(kl >= 0)
        assert jnp.all(hilbert >= 0)


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
        assert loss[-1, 0] < 0.9 * loss[0, 0]

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
        assert loss[-1, 0] < 0.9 * loss[0, 0]

    def test_train_on_emissions_reduces_loss(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=2)
        loss = rnn.train(
            casino,
            loss=LossType.EMISSIONS,
            batch_size=10,
            time_steps=50,
            num_epochs=1,
            optimization_steps=1000,
            print_every=999,
        )
        assert loss[-1, 0] < loss[0, 0]


class TestConstraints:
    def test_stochastic_constraint_enforced_after_training(self, casino):
        class StochasticModel(AbstractRNN):
            @staticmethod
            def schema(latent_dim, emission_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "stochastic"},
                    "B": {"shape": (latent_dim, emission_dim)},
                    "C": {"shape": (emission_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = StochasticModel(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.train(casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=999)
        A = np.array(rnn.get_parameter_values({"A"})["A"])
        assert np.allclose(A.sum(axis=0), 1.0, atol=1e-6)
        assert np.all(A >= 0)

    def test_stable_constraint_enforced_after_training(self, casino):
        class StableModel(AbstractRNN):
            @staticmethod
            def schema(latent_dim, emission_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "stable"},
                    "B": {"shape": (latent_dim, emission_dim)},
                    "C": {"shape": (emission_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = StableModel(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.train(casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=999)
        A = np.array(rnn.get_parameter_values({"A"})["A"])
        assert np.all(np.abs(np.linalg.eigvals(A)) <= 1 + 1e-5)

    def test_nonnegative_constraint_enforced_after_training(self, casino):
        class NonnegativeModel(AbstractRNN):
            @staticmethod
            def schema(latent_dim, emission_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "nonnegative"},
                    "B": {"shape": (latent_dim, emission_dim)},
                    "C": {"shape": (emission_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = NonnegativeModel(casino.latent_dim, casino.emission_dim, seed=0)
        rnn.train(casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=999)
        A = np.array(rnn.get_parameter_values({"A"})["A"])
        assert np.all(A >= 0)


class TestRegisterParameterType:
    def test_custom_parameter_usable_in_schema(self, casino):
        class AbsParameter(Parameter):
            """Parameter constrained to be element-wise non-negative via abs."""

            def get_value(self):
                return jnp.abs(self.dof)

        register_parameter_type("abs", AbsParameter)

        class AbsModel(AbstractRNN):
            @staticmethod
            def schema(latent_dim, emission_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "abs"},
                    "B": {"shape": (latent_dim, emission_dim)},
                    "C": {"shape": (emission_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = AbsModel(casino.latent_dim, casino.emission_dim, seed=0)
        assert isinstance(rnn._parameters["A"], AbsParameter)
        Y, X = rnn.predict(casino.sample(batch_size=2, time_steps=10)[1])
        assert Y.shape == (2, 10, casino.emission_dim)

    def test_register_rejects_non_parameter_subclass(self):
        with pytest.raises(TypeError):
            register_parameter_type("bad", object)

    def test_register_rejects_builtin_name(self):
        with pytest.raises(ValueError):
            register_parameter_type("stable", Parameter)


class TestFreezeUnfreeze:
    def test_freeze_prevents_update(self, casino):
        rnn = ModelA(casino.latent_dim, casino.emission_dim, seed=3)
        all_names = rnn.get_parameter_names()
        frozen = {"B", "C"}
        unfrozen = all_names - frozen
        rnn.freeze(frozen)
        before = rnn.get_parameter_values(all_names)
        rnn.train(
            casino,
            batch_size=5,
            time_steps=20,
            optimization_steps=10,
            print_every=999,
        )
        after = rnn.get_parameter_values(all_names)
        for name in frozen:
            assert np.allclose(before[name], after[name])
        for name in unfrozen:
            assert not np.allclose(before[name], after[name])
