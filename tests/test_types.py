"""Tests for enum Type handling."""

import pytest

from linear_rnn_filtering.types import ConstraintType, LossType


class TestConstruction:
    def test_instantiate_constraint_from_string(self):
        for member in ConstraintType:
            ConstraintType(member.value)

    def test_constraint_value_checking(self):
        with pytest.raises(ValueError):
            ConstraintType("unknown_type")

    def test_instantiate_loss_from_string(self):
        for member in LossType:
            LossType(member.value)

    def test_loss_value_checking(self):
        with pytest.raises(ValueError):
            LossType("unknown_type")
