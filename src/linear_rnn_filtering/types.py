"""Enum types used throughout the library for loss selection and weight constraints."""

from enum import Enum

__all__ = ["LossType", "ConstraintType"]


class LossType(str, Enum):
    """Training / evaluation objective."""

    EMISSIONS = "emissions"
    """Next-token negative log-likelihood."""
    KL = "kl"
    """KL divergence to ground-truth posterior."""
    HILBERT = "hilbert"
    """Hilbert projective distance to posterior."""


class ConstraintType(str, Enum):
    """Constraints on network weights."""

    UNCONSTRAINED = "unconstrained"
    """Default. No constraint."""
    STABLE = "stable"
    """All eigenvalues have modulus less than or equal to one."""
    STOCHASTIC = "stochastic"
    """Non-negative and columns sum to one."""
    NONNEGATIVE = "nonnegative"
    """Non-negative."""
