"""Enum types used throughout the library for loss selection and weight constraints."""

from enum import Enum
from typing import Any

__all__ = ["LossType", "ConstraintType", "Schema"]

Schema = dict[str, dict[str, Any]]


class CheckedType(str, Enum):
    """A Type class that raises a value error if initialized with an unrecognized value."""

    @classmethod
    def _missing_(cls, value):
        valid_values = [member.value for member in cls]
        raise ValueError(f"'{value}' is not a valid {cls.__name__}. Accepted values are: {valid_values}")


class LossType(CheckedType):
    """Training / evaluation objective."""

    EMISSIONS = "emissions"
    """Next-token negative log-likelihood."""
    KL = "kl"
    """KL divergence to ground-truth posterior."""
    HILBERT = "hilbert"
    """Hilbert projective distance to posterior."""


class ConstraintType(CheckedType):
    """Constraints on network weights."""

    UNCONSTRAINED = "unconstrained"
    """Default. No constraint."""
    STABLE = "stable"
    """All eigenvalues have modulus less than or equal to one."""
    STOCHASTIC = "stochastic"
    """Non-negative and columns sum to one."""
    NONNEGATIVE = "nonnegative"
    """Non-negative."""
