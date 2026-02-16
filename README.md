# linear-rnn-filtering

Linear RNN approximations to discrete Hidden Markov Model forward filtering.

This package provides:

- **`DiscreteHMM`** &mdash; Simulate a discrete HMM, sample hidden/emission trajectories in batch, and compute the exact Bayesian next-token posterior via forward filtering.
- **`ExactRNN`** &mdash; The exact nonlinear forward-filter recurrence implemented as a JAX-compatible RNN.
- **`ModelA`** &mdash; A stable linear RNN (`x_t = A x_{t-1} + B[:, y_t]`) with Cayley-parameterised dynamics and column-stochastic readout. Supports `initialize_Astar` for Jacobian-linearised initialization.
- **`ModelB`** &mdash; A stable linear RNN with an affine softmax readout (`p_t = softmax(C x_t + d)`).

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from linear_rnn_filtering import HMMFactory, ModelA, ExactRNN

# Create a two-state "dishonest casino" HMM
hmm = HMMFactory.dishonest_casino()

# Sample emission sequences
latent, emissions = hmm.sample(batch_size=100, time_steps=500)

# Compute the ground-truth Bayesian posterior
latent_posterior, next_token_posterior = hmm.compute_posterior(emissions)

# Create a Model A RNN and train it to match the posterior
rnn = ModelA(hmm.latent_dim, hmm.emission_dim, seed=0)
loss = rnn.train_on_posterior(hmm, batch_size=100, time_steps=500, optimization_steps=1000)

# Predict
Y, X = rnn.predict(emissions)

# Initialize an Astar model from HMM
rnn_lin = ModelA(hmm.latent_dim, hmm.emission_dim)
rnn_lin.initialize_Astar(hmm)

# Use initial condition for exact match (no burn-in needed)
import numpy as np
x0 = np.log(hmm.latent_stationary_density)
exact = ExactRNN(hmm.latent_dim, hmm.emission_dim)
exact.initialize_weights(hmm)
Y_exact, _ = exact.predict(emissions, x0=x0)
```

## Architecture summary

| Model | Dynamics | Readout | Schema |
|-------|----------|---------|--------|
| `ExactRNN` | `x_t = log(A exp(x_{t-1})) + B[:, y_t]` | `C @ softmax(x_t)` | A (stochastic), B (unconstrained), C (stochastic) |
| `ModelA` | `x_t = A @ x_{t-1} + B[:, y_t]` | `C @ softmax(x_t)` | A (stable/Cayley), B (unconstrained), C (stochastic) |
| `ModelB` | `x_t = A @ x_{t-1} + B[:, y_t]` | `softmax(C @ x_t + d)` | A (stable/Cayley), B, C, d (unconstrained) |

## Defining new models

Subclass `AbstractRNN` with a `schema` classmethod and an `integrate` static method:

```python
from linear_rnn_filtering import AbstractRNN

class MyModel(AbstractRNN):
    @classmethod
    def schema(cls, n, m):
        return [
            ("A", (n, n), "stable"),        # Cayley-parameterised
            ("B", (n, m), "unconstrained"),  # free parameters
            ("C", (m, n), "stochastic"),     # softmax-normalised columns
        ]

    @staticmethod
    def integrate(A, B, C, x_prev, emission_t):
        x_t = A @ x_prev + B[:, emission_t]
        y_t = C @ jax.nn.softmax(x_t+d)
        return x_t, y_t
```

Constraint types: `"unconstrained"`, `"stable"` (Cayley), `"stochastic"` (softmax axis=0), `"nonneg"` (squared).

## Training modes

All models support three training objectives:

- **`LossType.EMISSIONS`** &mdash; Minimise mean surprisal (negative log-likelihood of next token).
- **`LossType.KL`** &mdash; Minimise KL divergence to the exact HMM next-token posterior.
- **`LossType.HILBERT`** &mdash; Minimise the Hilbert projective metric to the exact HMM next-token posterior.

All accept an optional `x0` argument for initial hidden state.

## Development

```bash
git clone https://github.com/jpughesanford/linear-rnn-filtering
cd linear-rnn-filtering
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

MIT
