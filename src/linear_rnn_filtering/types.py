"""Enum types used throughout the library for loss selection and weight constraints."""

from enum import Enum

__all__ = ["LossType", "ConstraintType"]


class LossType(str, Enum):
    """Training / evaluation objective.

    Attributes:
        EMISSIONS: Next-token negative log-likelihood.
        KL: KL divergence to ground-truth posterior.
        HILBERT: Hilbert projective distance to posterior.
    """

    EMISSIONS = "emissions"
    KL = "kl"
    HILBERT = "hilbert"


class ConstraintType(str, Enum):
    """Constraints on network weights.

    Attributes:
        UNCONSTRAINED: Default. No constraint.
        STABLE: all eigenvalues have modulus less than or equal to one
        STOCHASTIC: Non-negative and columns sum to one.
        NONNEGATIVE: Non-negative.
    """

    UNCONSTRAINED = "unconstrained"
    STABLE = "stable"
    STOCHASTIC = "stochastic"
    NONNEGATIVE = "nonnegative"
