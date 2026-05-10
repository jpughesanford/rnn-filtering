"""RNN submodule for sequence modelling over vector-valued inputs."""

from .abstract import AbstractRNN
from .models import ExactRNN, LinearRNN
from .parameters import Parameter, register_parameter_type
from .types import ConstraintType

__all__ = [
    "AbstractRNN",
    "ExactRNN",
    "LinearRNN",
    "Parameter",
    "register_parameter_type",
    "ConstraintType"
]
