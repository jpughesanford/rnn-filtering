"""HMM and RNN module for studying RNN approximations to Bayesian forward filtering of
discrete time, discrete space HMMs. Implemented in JAX."""

import jax

jax.config.update("jax_enable_x64", True)

from . import hmm, rnn
from .training import train_on_hmm

__all__ = ["hmm", "rnn", "train_on_hmm"]

__version__ = "1.2.0"
