"""submodule for training RNNs. Includes helper functions for training RNNs on HMMs."""

from .loss_functions import hilbert_distance, kl_divergence, one_norm
from .utils import train

__all__ = [
    "train",
    "kl_divergence",
    "hilbert_distance",
    "one_norm",
]
