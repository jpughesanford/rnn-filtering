# linear-rnn-filtering

This repository provides the base code that was used in [submitted manuscript] to
investigate the ability of Recurrent Neural Networks (RNNs) with linear latent dynamics to
perform next token prediction on sequences sampled from Hidden Markov Models (HMMs).

At the moment, this code does not reproduce the datasets/figures/analysis done in the manuscript, but that is something
we plan to implement soon. What this version of the repo does is make public the JAX-based, fully differentiable, HMM
and RNN classes that we developed for this project.

## Installation

You can install this package into your local environment using pip:
```bash
pip install git+https://github.com/jpughesanford/linear-rnn-filtering.git
```

## Repository structure

The library is designed around two primary, top-level modules:

- **`linear_rnn_filtering.hmm`** — Discrete HMM simulation and exact Bayesian inference.
  Includes `DiscreteHMM` and `HMMFactory` for constructing standard HMMs.

- **`linear_rnn_filtering.rnn`** — RNN architectures for approximating HMM forward filtering.
  Includes `AbstractRNN`, `ExactRNN`, `ModelA`, and `ModelB`. This submodule allows the user to easily design their own constrained RNN architectures as well. 

Additionally, the `rnn.parameters` submodule exposes the `Parameter` base class (see below for more about parameters), along with a `register_parameter_type` method for user-defined constraints.

Shared enum types (`LossType`, `ConstraintType`) and the `Schema` type alias live in
`linear_rnn_filtering.types`.

## Quick start

```python
import numpy as np
from linear_rnn_filtering.hmm import HMMFactory
from linear_rnn_filtering.rnn import ModelA, ExactRNN

# Create a two-state "dishonest casino" HMM
hmm = HMMFactory.dishonest_casino()

# Sample 100 different emission sequences, each 500 symbols long
latent, emissions = hmm.sample(batch_size=100, time_steps=500)

# Compute the ground-truth, next-token Bayesian posterior over each of the 100 sequences
latent_posterior, next_token_posterior = hmm.compute_posterior(emissions)

# Create a Model A RNN and train it to match the posterior. This is an RNN with linear latent dynamics and a nonlinear readout function. 
# See manuscript for more details
rnn = ModelA(hmm.latent_dim, hmm.emission_dim, seed=0)
loss = rnn.train(hmm, loss="kl", batch_size=100, time_steps=500, optimization_steps=1000)

# run the RNN, forcing it with the emission sequences. Its output will approximate the true next token posterior
output_timeseries, latent_timeseries = rnn.predict(emissions)

# Implement an exact RNN. This is an RNN with nonlinear latent dynamics and a nonlinear readout function. 
# See manuscript for more details
exact = ExactRNN(hmm.latent_dim, hmm.emission_dim)

# For the right choice of weights, this RNN can perform forward filtering exactly. These weights can be 
# analytically solved for, and we wrote a method that sets them analytically, directly from the hmm instance:
exact.initialize_weights(hmm)

# run the RNN, forcing it with the emission sequences. Its output will equal the true next token posterior
x0 = np.log(hmm.latent_stationary_density)
exact_output_timeseries, _ = exact.predict(emissions, x0=x0)
```

### Loss functions supported

All models support three training objectives:

- **`"emissions"`** &mdash; Minimise mean surprisal (negative log-likelihood of next token).
- **`"kl"`** &mdash; Minimise KL divergence to the exact HMM next-token posterior.
- **`"hilbert"`** &mdash; Minimise the Hilbert projective metric to the exact HMM next-token posterior.

All loss functions accept an optional `x0` argument for the initial hidden state. `x0` defaults to all zeros. 


## Architecture summary

Here `latent_t` denotes the hidden state vector and `readout_t` the predicted next-token probability distribution.

| Model | Integrates                                               | Readout                          | Constraints                                   |
|-------|----------------------------------------------------------|----------------------------------------|-----------------------------------------------|
| `ExactRNN` | `latent_t = log(A exp(latent_{t-1})) + B[:, emission_t]` | `readout_t = C @ softmax(latent_t)`    | `A`, `C` stochastic; `B` unconstrained         |
| `ModelA` | `latent_t = A @ latent_{t-1} + B[:, emission_t]`         | `readout_t = C @ softmax(latent_t)`    | `A` Schur stable; `C` stochastic; `B` unconstrained |
| `ModelB` | `latent_t = A @ latent_{t-1} + B[:, emission_t]`         | `readout_t = softmax(C @ latent_t + d)` | `A` Schur stable; `B`, `C`, `d` unconstrained         |

## Parameters and constraints

### The Parameter pattern

Every named weight in a schema is backed by a `Parameter` instance. The key idea is that
each `Parameter` separates two things:

- **`dof`** — the unconstrained *degrees of freedom* that JAX is free to optimize.
- **`get_value()`** — the constrained *value* of the parameter that is passed to the integrate and readout steps. A parameters value is computed as a differentiable function of `dof`.

As an example, consider wanting a weight matrix, `Y`, to be constrained to be a stochastic matrix. This is what is implemented by the `StochasticParameter` class. 
The `dof` of a `StochasticParameter` is an unconstrained array `X`. When asked for its value, `StochasticParameter` returns 
`Y = softmax(X,axis=0)`. JAX varies `X` throughout training, and the RNNs `integrate` method only sees the parameters value: `Y`, a stochastic matrix 

This means constraints are always enforced exactly throughout training,
and in a way that remains fully JAX-differentiable.

You can inspect or set parameter values by name at any time:

```python
values = rnn.get_parameter_values({"A", "B", "C"})  # returns a dict of parameter values
rnn.set_parameter_values({"A": my_new_value})       # sets via the inverse of get_value

rnn.get_parameter_values({"A"})                     # now returns {"A": my_new_value}
```
To reiterate: `my_new_value` is the desired *value* of the parameter, not the desired state of the degrees of freedom. 

Individual parameters can also be frozen so that their `dof` are held fixed during training:

```python
rnn.freeze({"B", "C"})   # fix B and C
rnn.train(hmm, ...)       # only A is updated
rnn.unfreeze({"B", "C"})
```

### Built-in constraints

Our code allows you to constrain any weight parameter using the following built-in types:

| Key               | Guarantee                                                                                                                                                                   | Parameterization                                                                |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `"unconstrained"` | None                                                                                                                                                                        | `X = dof`                                                                       |
| `"stable"`        | Spectral radius ≤ 1 (Schur stable). Square matrices only. All stable matrices representable except those with eigenvalue exactly `−1`. See manuscript appendix for details. | `value = cayley(Y @ Y.T + ε I + skew(Z))` where `dof = (Y,Z)` are unconstrained |
| `"stochastic"`    | Columns non-negative and sum to one                                                                                                                                         | `value = softmax(dof, axis=0)` where `dof` is unconstrained                     |
| `"nonnegative"`   | All entries non-negative                                                                                                                                                    | `X = dof*dof` where `dof` is unconstrained                                      |

Mixing multiple constraint types on a single parameter is not supported.

### Registering custom constraints

If the built-in constraints don't cover your needs, you can register your own. Subclass
`Parameter`, override `get_value()` (and `set_value()` as needed), then register it under
a string key. That key can then be used in any schema dict exactly like the built-in names.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from linear_rnn_filtering import rnn

class BoundedParameter(rnn.parameters.Parameter):
    """Constrains all entries to the open interval (0, 1) via the sigmoid function."""

    def get_value(self):
        return jax.nn.sigmoid(self.dof)

    def set_value(self, value):
        value = jnp.asarray(value)
        if value.shape != self.shape:
            raise ValueError(f"Value shape {value.shape} does not match parameter shape {self.shape}.")
        if jnp.any(value <= 0) or jnp.any(value >= 1):
            raise ValueError("BoundedParameter value must lie in the open interval (0, 1).")
        return eqx.tree_at(lambda s: s.dof, self, jnp.log(value / (1 - value)))

rnn.parameters.register_parameter_type("bounded", BoundedParameter)
```

`eqx.tree_at` appears here because `Parameter` extends `equinox.Module`, which is immutable — you cannot modify fields in place. `eqx.tree_at` is Equinox's standard way to return a new copy of a module with one field replaced, and is the correct pattern for implementing `set_value` in any `Parameter` subclass.

After registration, `"bounded"` is available as a constraint in any schema:

```python
class MyModel(AbstractRNN):
    @staticmethod
    def schema(latent_dim, emission_dim):
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": "stable"},
            "alpha": {"shape": (latent_dim,), "constraint": "bounded"},
        }
```

## Defining new models

When starting this academic venture, we did not know specifically what model architectures we wanted to explore.
Consequently, we wrote code that made prototyping different architectures very efficient. To write your own RNN,
with whatever latent dynamics and non-linear readout you want, subclass `AbstractRNN` with `schema` and `integrate`
static methods: (below, we use `sin` and `cos` just to express that evolution can be nonlinear)

```python
from linear_rnn_filtering.rnn import AbstractRNN
import jax.numpy as jnp

class MyNonlinearModel(AbstractRNN):
    @staticmethod
    def schema(latent_dim, emission_dim):
        return {
            "A": {
                "shape": (latent_dim, latent_dim),
                "constraint": "stable"
            },
            "B": {
                "shape": (latent_dim, emission_dim),
            },
            "C": {
                "shape": (emission_dim, latent_dim),
                "constraint": "stochastic"
            }
        }

    @staticmethod
    def integrate(A, B, C, x_prev, emission_t):
        x_t = A @ jnp.sin(x_prev) + B[:, emission_t]
        y_t = C @ jnp.cos(x_t)
        return x_t, y_t

# for some hmm...
rnn = MyNonlinearModel(hmm.latent_dim, hmm.emission_dim, seed=0)
loss = rnn.train(hmm, loss="kl", batch_size=100, time_steps=500, optimization_steps=1000)
```

Recall that `integrate` has to function with JAX arrays.

Each key in the schema dict corresponds to an argument name in `integrate`. The `shape` field
defaults to `(1,)` and `constraint` defaults to `"unconstrained"` if omitted. You can also
pass `"initial_value"` to set a weight to a specific starting value rather than drawing it
randomly.

## License

MIT
