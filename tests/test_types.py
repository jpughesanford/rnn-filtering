"""Tests for enum Type handling."""

import pytest

from rnn_filtering.rnn.types import ConstraintType


class TestConstruction:
    def test_instantiate_constraint_from_string(self):
        for member in ConstraintType:
            ConstraintType(member.value)

    def test_constraint_value_checking(self):
        with pytest.raises(ValueError):
            ConstraintType("unknown_type")
