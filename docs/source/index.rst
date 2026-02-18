Linear RNN Filtering
====================

This repository provides the base code that was used in [submitted manuscript] to
investigate the ability of Recurrent Neural Networks (RNNs) with linear latent dynamics to
perform next token prediction on sequences sampled from Hidden Markov Models (HMMs).

For more information on this repository, please consult the :doc:`README`.

The repository is structured into three submodules: rnn, hmm, and types. Types holds
general enums, like those used to define supported loss-functions and supported constraints
on training variables.

The hmm modules contains classes:

- **DiscreteHMM**: Simulate a discrete HMM, sample hidden/emission trajectories in batch, and compute the exact Bayesian next-token posterior via forward filtering.
- **HMMFactory**: A factory pattern class, used to quickly instantiate common HMM types

The rnn module contains classes:

- **AbstractRNN**: Abstract base class to be subclassed when implementing arbitrary, potentially nonlinear, single-layer RNNs.
- **ExactRNN**: The exact nonlinear forward-filter implemented as an RNN.
- **ModelA**: A stable linear RNN, with a forward-filter informed nonlinear readout. Supports creating A* models (i.e., Jacobian-linearized initialization) using :meth:`initialize_Astar`, see manuscript for more details on such models.
- **ModelB**: A stable linear RNN, with an affine softmax readout (`output = softmax(A*latent_state + b)`).

API Reference
=============

linear_rnn_filtering.hmm
------------------------

.. autosummary::
   :toctree: _autosummary/hmm
   :caption: Hidden Markov Models

   ~linear_rnn_filtering.hmm.DiscreteHMM
   ~linear_rnn_filtering.hmm.HMMFactory


linear_rnn_filtering.rnn
------------------------

.. autosummary::
   :toctree: _autosummary/rnn
   :caption: RNN models

   ~linear_rnn_filtering.rnn.AbstractRNN
   ~linear_rnn_filtering.rnn.ExactRNN
   ~linear_rnn_filtering.rnn.ModelA
   ~linear_rnn_filtering.rnn.ModelB


linear_rnn_filtering.types
--------------------------

.. autosummary::
   :toctree: _autosummary/types
   :caption: Enums and type definitions

   ~linear_rnn_filtering.types.LossType
   ~linear_rnn_filtering.types.ConstraintType