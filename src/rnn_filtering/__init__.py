"""HMM and RNN module for studying RNN approximations to Bayesian forward filtering of
discrete time, discrete space HMMs. Implemented in JAX."""

import jax

jax.config.update("jax_enable_x64", True)

from . import hmm, rnn

__all__ = ["hmm", "rnn"]

__version__ = "1.2.0"
