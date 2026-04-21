import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rnn_filtering.hmm import HMMFactory, NodeEmittingHMM
from rnn_filtering.rnn import AbstractRNN, Parameter, register_parameter_type
from rnn_filtering.rnn.parameters import StableParameter
from rnn_filtering.training import train_on_hmm


@pytest.fixture
def casino():
    return HMMFactory.dishonest_casino()


class TestConstraints:
    def test_stochastic_constraint_enforced_after_training(self, casino):
        class StochasticModel(AbstractRNN):
            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "stochastic"},
                    "B": {"shape": (latent_dim, input_dim)},
                    "C": {"shape": (output_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = StochasticModel(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        train_on_hmm(rnn, casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=0)
        A = np.array(rnn.get_parameter_values({"A"})["A"])
        assert np.allclose(A.sum(axis=0), 1.0, atol=1e-6)
        assert np.all(A >= 0)

    def test_stable_constraint_enforced_after_training(self, casino):
        class StableModel(AbstractRNN):
            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "stable"},
                    "B": {"shape": (latent_dim, input_dim)},
                    "C": {"shape": (output_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = StableModel(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        train_on_hmm(rnn, casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=0)
        A = np.array(rnn.get_parameter_values({"A"})["A"])
        assert np.all(np.abs(np.linalg.eigvals(A)) <= 1 + 1e-5)

    def test_nonnegative_constraint_enforced_after_training(self, casino):
        class NonnegativeModel(AbstractRNN):
            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "nonnegative"},
                    "B": {"shape": (latent_dim, input_dim)},
                    "C": {"shape": (output_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = NonnegativeModel(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        train_on_hmm(rnn, casino, batch_size=10, time_steps=50, optimization_steps=50, print_every=0)
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
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "abs"},
                    "B": {"shape": (latent_dim, input_dim)},
                    "C": {"shape": (output_dim, latent_dim)},
                }

            @staticmethod
            def integrate(A, B, C, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = jax.nn.softmax(C @ x_t)
                return x_t, y_t

        rnn = AbsModel(casino.emission_dim, casino.latent_dim, casino.emission_dim, seed=0)
        assert isinstance(rnn._parameters["A"], AbsParameter)
        _, emissions = casino.sample(batch_size=2, time_steps=10)
        inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), casino.emission_dim)
        Y, X = rnn.respond(inputs)
        assert Y.shape == (2, 10, casino.emission_dim)

    def test_register_rejects_non_parameter_subclass(self):
        with pytest.raises(TypeError):
            register_parameter_type("bad", object)

    def test_register_rejects_builtin_name(self):
        with pytest.raises(ValueError):
            register_parameter_type("stable", Parameter)

    def test_lifecycle(self):
        class ZeroColumnSumMatrix(Parameter):
            """Represents a matrix A whose columns sum to 0: {\bf 1}^\top A = 0."""

            def get_value(self):
                """Return A, where the last row is computed as the negative column sum of the previous n-1 rows."""
                last_row = -jnp.sum(self.dof, axis=0, keepdims=True)
                return jnp.vstack([self.dof, last_row])

            def set_value(self, value):
                """store only the top n-1 rows of A"""
                value = jnp.asarray(value)
                return eqx.tree_at(lambda s: s.dof, self, value[:-1, :])

        register_parameter_type("zero_column_sum", ZeroColumnSumMatrix)

        class RNN(AbstractRNN):
            @staticmethod
            def schema(input_dim, latent_dim, output_dim):
                return {
                    "A": {"shape": (latent_dim, latent_dim), "constraint": "zero_column_sum"},
                    "B": {"shape": (latent_dim, input_dim), "constraint": "zero_column_sum"},
                }

            @staticmethod
            def integrate(A, B, x_prev, input_t):
                x_t = A @ x_prev + B @ input_t
                y_t = x_t
                return x_t, y_t

        E = np.asarray([[3, 1], [1, 3]]) / 4
        T = np.asarray([[1 - 0.1, 0.1], [0.1, 1 - 0.1]])
        hmm = NodeEmittingHMM(transfer_operator=T, emission_operator=E, latent_dim=2, emission_dim=2)
        rnn = RNN(hmm.emission_dim, hmm.latent_dim, hmm.emission_dim, seed=0)
        params = rnn.get_parameter_values(["A", "B"])
        assert np.all(params["A"].sum(axis=0) == 0)
        assert np.all(params["B"].sum(axis=0) == 0)


class TestStableMatrixParametrization:
    """Tests for StableParameter.stable_matrix_to_params / params_to_stable_matrix round-trip."""

    # A matrix provably in the image of params_to_stable_matrix (generated by the forward map).
    A_ref = jnp.array([[0.832299, 0.034902], [-0.139604, 0.902103]])
    A1_ref = jnp.array([[0.3, 0.0], [0.1, 0.2]])
    A2_ref = jnp.array([0.1])

    # Casino Jacobian at the stationary distribution: eigenvalues 1.0 and 0.85.
    A_marginal = np.array([[0.95, 0.05], [0.10, 0.90]])

    # Matrix with eigenvalue exactly -1: not representable by the Cayley parameterization.
    A_neg_one = np.diag([-1.0, 0.5])

    def test_inverse(self):
        A_num = StableParameter.params_to_stable_matrix(self.A1_ref, self.A2_ref)
        A1, A2 = StableParameter.stable_matrix_to_params(A_num)
        assert np.allclose(A1, self.A1_ref, atol=1e-5)
        assert np.allclose(A2, self.A2_ref, atol=1e-5)

        A1, A2 = StableParameter.stable_matrix_to_params(self.A_ref)
        A = StableParameter.params_to_stable_matrix(A1, A2)
        assert np.allclose(A, self.A_ref, atol=1e-5)

    def test_marginal_stable_matrix(self):
        """Marginally stable matrices (eigenvalue +1) must not produce NaN in the round-trip."""
        A1, A2 = StableParameter.stable_matrix_to_params(jnp.asarray(self.A_marginal))
        A_recovered = np.array(StableParameter.params_to_stable_matrix(A1, A2))
        assert not np.any(np.isnan(A_recovered))
        assert np.all(np.abs(np.linalg.eigvals(A_recovered)) <= 1)

    def test_neg_one_eigenvalue_raises_error(self):
        """Matrices with eigenvalue -1 are not representable and must raise ValueError."""
        with pytest.raises(ValueError, match="-1"):
            StableParameter.stable_matrix_to_params(jnp.asarray(self.A_neg_one))
