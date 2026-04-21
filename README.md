# rnn-filtering


A JAX library for Hidden Markov Model (HMM) simulation and forward filtering + Recurrent Neural Network (RNN) filtering approximation.

This repository contains the framework developed for investigating how RNNs learn to perform Bayesian inference. It provides a bridge between probabilistic graphical models and deep learning architectures, featuring a unique differentiable constraint system for weights.

### Key Features

* **Exact Inference:** JAX-based HMMs for sampling and exact Bayesian filtering.
* **Filtering RNNs:** Architectures (Model A, B, and Exact) designed to approximate HMM forward filtering.
* **Differentiable Constraints:** A `Parameter` system that enforces spectral stability or stochasticity while remaining fully JAX-differentiable.
* **JAX-Native:** Fully differentiable, vectorized, and hardware-accelerated.

## Installation

Install the package into your local environment using pip:
```bash
pip install git+https://github.com/jpughesanford/rnn-filtering.git
```

## Repository structure

The library is designed around two primary, top-level modules plus a top-level training helper method:

- **`rnn_filtering.hmm`** — Discrete HMM simulation and exact Bayesian inference.
  Includes `AbstractHMM`, `NodeEmittingHMM`, `EdgeEmittingHMM`, and `HMMFactory`.

- **`rnn_filtering.rnn`** — General-purpose RNN architectures over vector-valued inputs.
  Includes `AbstractRNN`, `ExactRNN`, `ModelA`, and `ModelB`. This submodule allows the user to easily design their own constrained RNN architectures as well. The RNN class is largely HMM-agnostic; it operates on arbitrary sequences of fixed-length vectors.

- **`rnn_filtering.train_on_hmm`** — Convenience wrapper that couples the RNN training loop to an HMM: samples emission sequences, embeds them as one-hot vectors, computes posteriors, and loops over epochs.

Additionally, the `rnn` submodule exposes the `Parameter` base class (see below for more about parameters), along with a `register_parameter_type` method for user-defined constraints.

## Quick start

The following example demonstrates the "Distillation" workflow: using an HMM as a teacher to generate data and an RNN as a student to approximate the posterior.

```python
import jax
import jax.numpy as jnp
import numpy as np
from rnn_filtering.hmm import HMMFactory
from rnn_filtering.rnn import ModelA, ExactRNN
from rnn_filtering import train_on_hmm

# Create a two-state "dishonest casino" HMM
hmm = HMMFactory.dishonest_casino()

# Sample 100 different emission sequences, each 500 symbols long
latent, emissions = hmm.sample(batch_size=100, time_steps=500)

# Compute the ground-truth, next-token Bayesian posterior over each of the 100 sequences
# latent_posterior: p(x_t | y_{1:t})
# next_token_posterior: p(y_{t+1} | y_{1:t})
latent_posterior, next_token_posterior = hmm.compute_posterior(emissions)

# Create a Model A RNN and train it to match the posterior.
# train_on_hmm handles sampling, one-hot embedding, and posterior computation.
rnn = ModelA(hmm.emission_dim, hmm.latent_dim, hmm.emission_dim, seed=0)
loss = train_on_hmm(rnn, hmm, output_loss="kl", batch_size=100, time_steps=500, optimization_steps=1000)

# Embed emissions as one-hot vectors and run the RNN.
inputs = jax.nn.one_hot(jnp.asarray(emissions, jnp.int32), hmm.emission_dim)
output_timeseries, latent_timeseries = rnn.respond(inputs)

# Implement an exact RNN. This is an RNN with nonlinear latent dynamics and a nonlinear readout function.
# See manuscript for more details
exact = ExactRNN(hmm.emission_dim, hmm.latent_dim, hmm.emission_dim)

# For the right choice of weights, this RNN can perform forward filtering exactly. These weights can be
# analytically solved for, and we wrote a method that sets them analytically, directly from the hmm instance:
exact.initialize_weights(hmm)

# Run the RNN; its output will equal the true next token posterior.
x0 = np.log(hmm.latent_stationary_density)
exact_output_timeseries, _ = exact.respond(inputs, x0=x0)
```

## HMM classes

Both HMM classes accept operators as either dense arrays or JAX-traceable functions, and expose the same `sample` and `compute_posterior` interface.

### NodeEmittingHMM

Emissions depend only on the current latent state: `P(y_t | x_t)`. Accepts a transfer operator and an emission operator as either dense arrays or callables.

Below is an example of an HMM being defined via matrices. Both matrices must be column-stochastic.
```python
from rnn_filtering.hmm import NodeEmittingHMM
import numpy as np

# Matrix form: transfer matrix shape (latent_dim, latent_dim), emission matrix shape (emission_dim, latent_dim).
# Dishonest casino: two latent states (fair die, loaded die), six emission symbols (die faces).
transfer_matrix = np.array([[0.75, 0.50],   # columns are P(next state | current state)
                             [0.25, 0.50]])
emission_matrix = np.array([[1/6, 1/10],    # rows are faces 1-6, columns are fair/loaded
                             [1/6, 1/10],
                             [1/6, 1/10],
                             [1/6, 1/10],
                             [1/6, 1/10],
                             [1/6, 1/2 ]])

hmm = NodeEmittingHMM(latent_dim=2, emission_dim=6, transfer_operator=transfer_matrix, emission_operator=emission_matrix)
```

Some HMMs are easy to describe as functions but cumbersome as matrices. For example, a symmetric random walk on a ring of n=10^6 nodes would require a 10^12-entry transition matrix. These can instead be specified as a pair of JAX-traceable functions:

```python
from rnn_filtering.hmm import NodeEmittingHMM
import jax.numpy as jnp

# Callable form: useful when the operator has natural structure.
n = 10**6

def transition_function(state):   # R^n -> R^n
    return 0.5 * jnp.roll(state, 1) + 0.5 * jnp.roll(state, -1)

def emission_function(state):     # R^n -> R^2
    return jnp.array([state[:n//2].sum(), state[n//2:].sum()])

hmm = NodeEmittingHMM(n, 2, transition_function, emission_function)
```

In either case, the stationary distribution is computed automatically — via eigendecomposition for matrix inputs, and Anderson acceleration for callables. If the stationary distribution is known, pass it to skip this computation:

```python
hmm = NodeEmittingHMM(2, 6, transfer_matrix_or_function, emission_matrix_or_function, latent_stationary_density=pi)
```

### EdgeEmittingHMM

Emissions depend on both the previous and current latent state (the transition edge): `P(y_t | x_{t}, x_{t-1})`. Accepts a transfer operator and an **edge** emission operator:

```python
from rnn_filtering.hmm import EdgeEmittingHMM

# ndarrary form: emission_ndarray shape (emission_dim, latent_dim, latent_dim), column-stochastic along axis 0.
# Axes are ordered from most to least recent: E[y_t, x_t, x_{t-1}].
hmm = EdgeEmittingHMM(latent_dim=2, emission_dim=6, transfer_operator=transfer_matrix, emission_operator=emission_ndarray)

# Callable form: arguments are ordered from most to least recent: f(x_t, x_{t-1}).
def transfer_function(current_state):   # R^n -> R^n
    return transfer_matrix @ current_state

def emission_function(current_state, prev_state):   # R^n x R^n -> R^m
    return (emission_ndarray @ prev_state) @ current_state

hmm = EdgeEmittingHMM(2, 6, transfer_matrix, emission_ndarray)
# or
hmm = EdgeEmittingHMM(2, 6, transfer_function, emission_function)
```

The index/argument ordering convention mirrors the transfer operator: the output comes first, followed by conditioning variables from most to least recent. For `E[y, j, k]`: $y$ is the emission at time $t$, $j = x_t$, $k = x_{t-1}$.

### HMMFactory

`HMMFactory` provides convenience constructors for standard HMMs:

```python
from rnn_filtering.hmm import HMMFactory

hmm = HMMFactory.dishonest_casino()               # classic 2-state node-emitting casino example
hmm = HMMFactory.random_dirichlet(latent_dim=4, emission_dim=8)  # random node emitting HMM with matrices samples from Dirichlet priors
hmm = HMMFactory.dyck_fun(depth=2, width=4)  # an edge-emitting, functionally defined Dyck language of depth 2 and width 4. dyck_arr implements the same hmm using matrices.
```

## RNN Classes

This repository exposes an easily extendable `AbstractRNN` class, capable of modelling a nonlinear, single-layer RNN. It also includes three pre-written RNN classes we use to explore HMM inference:

|  Model   |               Integrates               |               Readout               |                  Constraints                  |
|:--------:|:--------------------------------------:|:-----------------------------------:|:---------------------------------------------:|
| Exact RNN | $x_t = \log(A \exp(x_{t-1})) + B\ u_t$ |   $p_t = C\ \text{softmax}(x_t)$    |     `A`, `C` stochastic; `B` unconstrained     |
| Model A  |     $x_t = A\ x_{t-1} + B\ u_t$     |   $p_t = C\ \text{softmax}(x_t)$    | `A` Schur stable; `C` stochastic; `B` unconstrained |
| Model B  |    $x_t = A\  x_{t-1} + B\ u_t$     | $p_t = \text{softmax}(C \ x_t + d)$ |     `A` Schur stable; `B`, `C`, `d` unconstrained     |

Here $x_t$ denotes the hidden state vector, $u_t$ denotes the input vector at time $t$ (e.g. a one-hot embedding of an emission symbol), and $p_t$ is the output of the RNN (trained to approximate the next token posterior probability distribution $p_{t,i}=P(y_{t+1}=i| y_{0:t})$).

### Training

There are two methods for training an RNN:

- **`rnn.train(inputs, desired_output=..., output_loss=..., ...)`** — HMM-agnostic. Accepts a pre-computed batch of input vectors `(B, T, K)` and optional target arrays. Runs gradient descent for a fixed number of steps and returns a `(optimization_steps,)` loss history. Use this when you have your own data pipeline.

- **`train_on_hmm(rnn, hmm, output_loss=..., num_epochs=..., ...)`** — HMM-aware wrapper. Handles sampling, one-hot embedding of emission symbols, and posterior computation. Loops over `num_epochs` data-resampling epochs and returns a `(optimization_steps, num_epochs)` loss history.

### Loss functions

Both `rnn.train` and `train_on_hmm` accept a `output_loss` and an optional `latent_loss` argument. Each can be a string key or any callable with signature `(result, desired) -> scalar`.

Built-in string keys:

| Key | Description                                    | Target shape |
|-----|------------------------------------------------|--------------|
| `"kl"` | KL divergence KL(desired ‖ result)             | `(B, T, K)` distribution |
| `"hilbert"` | Hilbert projective metric                      | `(B, T, K)` distribution |
| `"one_norm"` | L1 norm between distributions                  | `(B, T, K)` distribution |
| `"emissions"` | One-hot KL = NLL (used in `train_on_hmm` only) | handled automatically |

`"emissions"` is only available through `train_on_hmm`, which converts integer emission symbols to one-hot vectors and passes them as targets to the KL loss. Calling `rnn.train` directly with `output_loss="emissions"` will raise an error — pass one-hot embeddings explicitly as `desired_output` with `output_loss="kl"` instead.

All built-in loss functions also support a `clip` keyword argument (default varies by function) and a `do_average` keyword argument (default `True`). To customise clipping, use `functools.partial`:

```python
from functools import partial
from rnn_filtering.rnn import kl_divergence

loss = train_on_hmm(rnn, hmm, output_loss=partial(kl_divergence, clip=1e-8))
```

### Parameters and constraints

A key challenge in training RNNs is ensuring that the learned weights remain physically or mathematically meaningful—for example, ensuring a transition matrix remains stochastic or a system remains stable. Our library addresses this through a **Parameter pattern** that decouples the optimization space from the model space. To JAX, a parameter appears as an unconstrained vector of degrees of freedom (**DOF**) that can be updated via gradient descent. To the RNN, this same parameter is automatically transformed into a valid, constrained **Value** (such as a Schur-stable or non-negative matrix) via a differentiable mapping. This approach guarantees that constraints are strictly enforced at every step of training without sacrificing the flexibility of JAX's autodiff engine.

#### The Parameter pattern

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
train_on_hmm(rnn, hmm, ...)   # only A is updated
rnn.unfreeze({"B", "C"})
```

#### Built-in constraints

Our code allows you to constrain any weight parameter using the following built-in types:

| Key               | Guarantee                                                                                                                                                                   | Parameterization                                                                |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `"unconstrained"` | None                                                                                                                                                                        | `value = dof`                                                                   |
| `"stable"`        | Spectral radius ≤ 1 (Schur stable). Square matrices only. All stable matrices representable except those with eigenvalue exactly `−1`. See manuscript appendix for details. | `value = cayley(Y @ Y.T + ε I + skew(Z))` where `dof = (Y,Z)` are unconstrained |
| `"stochastic"`    | Columns non-negative and sum to one                                                                                                                                         | `value = softmax(dof, axis=0)` where `dof` is unconstrained                     |
| `"nonnegative"`   | All entries non-negative                                                                                                                                                    | `value = dof*dof` where `dof` is unconstrained                                  |

Mixing multiple constraint types on a single parameter is not supported.

### Defining new models and constraints

When starting this academic venture, we did not know specifically what model architectures we wanted to explore.
Consequently, we wrote code that made prototyping different architectures very efficient. To write your own RNN,
with whatever latent dynamics and non-linear readout you want, subclass `AbstractRNN` with `schema` and `integrate`
static methods: (below, we use `sin` and `cos` just to express that evolution can be nonlinear)

```python
from rnn_filtering.rnn import AbstractRNN
from rnn_filtering import train_on_hmm
import jax.numpy as jnp

class MyNonlinearModel(AbstractRNN):
    @staticmethod
    def schema(input_dim, latent_dim, output_dim):
        return {
            "A": {
                "shape": (latent_dim, latent_dim),
                "constraint": "stable"
            },
            "B": {
                "shape": (latent_dim, input_dim),
            },
            "C": {
                "shape": (output_dim, latent_dim),
                "constraint": "stochastic"
            }
        }

    @staticmethod
    def integrate(A, B, C, x_prev, input_t):
        x_t = A @ jnp.sin(x_prev) + B @ input_t
        y_t = C @ jnp.cos(x_t)
        return x_t, y_t

# for some hmm...
rnn = MyNonlinearModel(hmm.latent_dim, hmm.emission_dim, hmm.emission_dim, seed=0)
loss = train_on_hmm(rnn, hmm, output_loss="kl", batch_size=100, time_steps=500, optimization_steps=1000)
```

Each key in the schema dict corresponds to an argument name in `integrate`. The `shape` field
defaults to `(1,)` and `constraint` defaults to `"unconstrained"` if omitted. You can also
pass `"initial_value"` to set a weight to a specific starting value rather than drawing it
randomly.

#### Registering custom constraints

If the built-in constraints don't cover your needs, you can register your own. Subclass
`Parameter`, override `get_value()` (and `set_value()` as needed), then register it under
a string key. That key can then be used in any schema dict exactly like the built-in names.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from rnn_filtering.rnn import Parameter, register_parameter_type

class BoundedParameter(Parameter):
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

register_parameter_type("bounded", BoundedParameter)
```

`eqx.tree_at` appears here because `Parameter` extends `equinox.Module`, which is immutable — you cannot modify fields in place. `eqx.tree_at` is Equinox's standard way to return a new copy of a module with one field replaced, and is the correct pattern for implementing `set_value` in any `Parameter` subclass.

After registration, `"bounded"` is available as a constraint in any schema:

```python
from rnn_filtering.rnn import AbstractRNN

class MyModel(AbstractRNN):
    @staticmethod
    def schema(input_dim, latent_dim, output_dim):
        return {
            "A": {"shape": (latent_dim, latent_dim), "constraint": "stable"},
            "alpha": {"shape": (latent_dim,), "constraint": "bounded"},
        }
    @staticmethod
    def integrate(A, alpha, x_prev, input_t):
        ...

my_rnn = MyModel(input_dim, latent_dim, output_dim)
```

## License

MIT
