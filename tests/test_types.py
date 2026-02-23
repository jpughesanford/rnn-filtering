"""Tests for enum Type handling."""

import jax.numpy as jnp
import numpy as np
import pytest

from linear_rnn_filtering.rnn import ExactRNN, AbstractRNN
from linear_rnn_filtering.types import LossType, ConstraintType


class TestConstruction:
    def test_instantiate_constraint_from_string(self):
        for member in ConstraintType:
            ConstraintType(member.value)

    def test_constraint_value_checking(self):
        with pytest.raises(ValueError) as e:
            ConstraintType("unknown_type")

    def test_instantiate_loss_from_string(self):
        for member in LossType:
            LossType(member.value)

    def test_loss_value_checking(self):
        with pytest.raises(ValueError) as e:
            LossType("unknown_type")



class TestUsage:
    def test_schema(self):
        class TestRNN(AbstractRNN):
            @staticmethod
            def schema(n, m):
                return [
                    ("A", (n, n), "stable"),
                    ("B", (n, m), ConstraintType.NONNEGATIVE),
                    ("C", (m, n), "nonnegative"),
                ]

            @staticmethod
            def integrate(A, B, C, x_prev, emission_t):
                x_t = A @ x_prev + B[:, emission_t]
                y_t = C @ x_t
                return x_t, y_t
        test = TestRNN(10,10)
        assert test._schema[0][2] == ConstraintType.STABLE
        assert test._schema[1][2] == ConstraintType.NONNEGATIVE
        assert test._schema[2][2] == ConstraintType.NONNEGATIVE