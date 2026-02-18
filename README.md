# README

This repository provides the base code that was used in [submitted manuscript] to
investigate the ability of Recurrent Neural Networks (RNNs) with linear latent dynamics to
perform next token prediction on sequences sampled from Hidden Markov Models (HMMs).

At the moment, this code does not reproduce the datasets/figures/analysis done in the manuscript, but that is something
we plan to implement soon. What this version of the repo does is make publish the JAX-based, fully differentiable, HMM
and RNN classes that we developed for this project. 

## Installation

```bash
# pip install 
pip install git+https://github.com/jpughesanford/linear-rnn-filtering.git   
```

## Quick start

```python
import numpy as np
from linear_rnn_filtering.hmm import HMMFactory
from linear_rnn_filtering.rnn import ModelA, ExactRNN
from linear_rnn_filtering.types import LossType

# Create a two-state "dishonest casino" HMM
hmm = HMMFactory.dishonest_casino()

# Sample 100 different emission sequences, each 500 symbols long
latent, emissions = hmm.sample(batch_size=100, time_steps=500)

# Compute the ground-truth, next-token Bayesian posterior over each of the 100 sequences
latent_posterior, next_token_posterior = hmm.compute_posterior(emissions)

# Create a Model A RNN and train it to match the posterior
rnn = ModelA(hmm.latent_dim, hmm.emission_dim, seed=0)
loss = rnn.train(hmm, loss=LossType.KL, batch_size=100, time_steps=500, optimization_steps=1000)

# run the RNN, forcing it with the emission sequences. Its output will approximate the true next token posterior
output_timeseries, latent_timeseries = rnn.predict(emissions)

# Implement an exact RNN. For the right choice of weights, this RNN can perform forward filtering exactly
exact = ExactRNN(hmm.latent_dim, hmm.emission_dim)

# Rather than train it, the ideal weights are analytically solvable. Set them using:
exact.initialize_weights(hmm)

# run the RNN, forcing it with the emission sequences. Its output will equal the true next token posterior
x0 = np.log(hmm.latent_stationary_density)
exact_output_timeseries, _ = exact.predict(emissions, x0=x0)
```

## Architecture summary

| Model | Dynamics                                                 | Readout                                 | Schema                                              |
|-------|----------------------------------------------------------|-----------------------------------------|-----------------------------------------------------|
| `ExactRNN` | `latent_t = log(A exp(latent_{t-1})) + B[:, emission_t]` | `readout_t = C @ softmax(latent_t)`     | A (stochastic), B (unconstrained), C (stochastic)   |
| `ModelA` | `latent_t = A @ latent_{t-1} + B[:, emission_t]`                | `readout_t = C @ softmax(latent_t)`     | A (Schur stable), B (unconstrained), C (stochastic) |
| `ModelB` | `latent_t = A @ latent_{t-1} + B[:, emission_t]`                | `readout_t = softmax(C @ latent_t + d)` | A (Schur stable), B, C, and d (unconstrained)       |

## Defining new models

When starting this academic venture, we did not know specifically what model architectures we wanted to explore. 
Consequently, we wrote code that made prototyping different architectures very efficient. To write your own RNN, 
with whatever latent dynamics and non-linear readout you want, subclass `AbstractRNN` with a `schema` classmethod and 
an `integrate`static method: (below, we use ``sin`` and ``cos`` just to express that evolution can be nonlinear)

```python
from linear_rnn_filtering.types import LossType, ConstraintType
from linear_rnn_filtering.rnn import AbstractRNN
import jax.numpy as jnp

class MyNonlinearModel(AbstractRNN):
    @classmethod
    def schema(cls, n, m):
        return [
            ("A", (n, n), ConstraintType.STABLE),        # Cayley-parameterised
            ("B", (n, m), ConstraintType.UNCONSTRAINED),  # free parameters
            ("C", (m, n), ConstraintType.NONNEGATIVE),     # softmax-normalised columns
        ]

    @staticmethod
    def integrate(A, B, C, x_prev, emission_t):
        x_t = A @ jnp.sin(x_prev) + B[:, emission_t]
        y_t = C @ jnp.cos(x_t)
        return x_t, y_t

# for some hmm...
rnn = MyNonlinearModel(hmm.latent_dim, hmm.emission_dim, seed=0)
loss = rnn.train(hmm, loss=LossType.KL, batch_size=100, time_steps=500, optimization_steps=1000)
```

Recall that integrate has to function with JAX arrays. 

Often, you want to constrain the weights, such that, for example, ``A`` is stable and the latent dynamics do not blow up. 
Our code allows you to constrain any weight parameter to be either:

- **`ConstraintType.UNCONSTRAINED`** &mdash; No constraints.
- **`ConstraintType.STABLE`** &mdash; Only enforcible for square matrices. Under the hood, the matrix ``X=f(Y,Z)`` is represented a function of two unconstrained matrices, ``Y``,``Z``. ``Y`` and ``Z`` are trained and ``X`` remains Schur stable through our choice of ``f``. All stable matrices, including non-normal ones, are representable by ``f``. However, matrices with ``-1`` as an eigenvalue are not representable by ``f``. See manuscript appendix for more details on ``f``. 
- **`ConstraintType.STOCHASTIC`** &mdash; Under the hood, the matrix ``X=softmax(Y,axis=0)``, where ``Y`` is an unconstrained matrix. 
- **`ConstraintType.NONNEGATIVE`** &mdash; Under the hood, the matrix ``X=Y**2``, where ``Y`` is an unconstrained matrix. 

Constraints will always be enforced throughout training, and in a way the remains fully JAX-differentiable. 

## Loss functions supported 

All models support three training objectives:

- **`LossType.EMISSIONS`** &mdash; Minimise mean surprisal (negative log-likelihood of next token).
- **`LossType.KL`** &mdash; Minimise KL divergence to the exact HMM next-token posterior.
- **`LossType.HILBERT`** &mdash; Minimise the Hilbert projective metric to the exact HMM next-token posterior.

All loss functions accept an optional `x0` argument for initial hidden state.

## License

MIT
