import jax
import jax.numpy as jnp
import numpy as np

from .models import EdgeEmittingHMM, NodeEmittingHMM


def _dyck_build_tree(depth: int, width: int):
    """Precompute analytic tree structure for a complete width-ary tree of given depth.

    Args:
        depth (int): Tree depth (root at depth 0, leaves at depth ``depth``).
        width (int): Branching factor.

    Returns:
        tuple: ``(num_latent_states, parent_of, which_child_of, is_leaf, is_root, children_arr, degree)`` where
            num_latent_states (int) is the number of nodes;
            parent_of (ndarray (num_latent_states,)) is the parent index with root mapping to 0 as a dummy sentinel;
            which_child_of (ndarray (num_latent_states,)) is the 0-indexed child slot in parent with root mapping to 0
                as a dummy sentinel;
            is_leaf (ndarray (num_latent_states,) bool) is True for nodes at depth ``depth``;
            is_root (ndarray (num_latent_states,) bool) is True for node 0 only;
            children_arr (ndarray (num_latent_states, width) int) holds ``i*width+m+1`` for valid children and `
                `num_latent_states`` as an out-of-bounds sentinel;
            degree (ndarray (num_latent_states,)) is the number of tree-adjacent neighbours: 1 (if non-root) + width
                (if non-leaf).
    """
    num_latent_states = depth + 1 if width == 1 else (width ** (depth + 1) - 1) // (width - 1)
    indices = np.arange(num_latent_states)

    parent_of = np.where(indices > 0, (indices - 1) // width, 0)
    which_child_of = np.where(indices > 0, (indices - 1) % width, 0)
    is_root = indices == 0
    is_leaf = indices * width + 1 >= num_latent_states
    children_raw = indices[:, None] * width + np.arange(width)[None, :] + 1
    children_arr = np.where(children_raw < num_latent_states, children_raw, num_latent_states)

    degree = np.zeros(num_latent_states)
    degree[~is_root] += 1.0
    degree[~is_leaf] += width

    return num_latent_states, parent_of, which_child_of, is_leaf, is_root, children_arr, degree


class HMMFactory:
    """HMM Factory for creating pre-configured ``HMM`` instances."""

    @staticmethod
    def dyck_arr(
        depth: int = 3,
        width: int = 2,
        temperature: float = 0.1,
    ) -> EdgeEmittingHMM:
        """Construct an array-backed Dyck language HMM on a complete width-ary tree.

        Suitable for small trees where storing the full N×N transfer matrix and
        (2*width+1)×N×N emission tensor is feasible. For large trees use ``dyck_fun``.

        The HMM has N = (width^(depth+1) - 1) / (width - 1) latent states.
        Transfer is a uniform random walk on the tree mixed with temperature-scaled
        teleportation. Emission encodes the bracket type of the edge traversed:

        - Symbol 0: epsilon (teleportation to a non-adjacent node).
        - Symbol 2m-1 (m=1..width): opening bracket m (descend to child m).
        - Symbol 2m   (m=1..width): closing bracket m (ascend from child m).

        Args:
            depth (int): Tree depth. Defaults to 3.
            width (int): Branching factor. Defaults to 2.
            temperature (float): Teleportation probability (0 = pure walk, 1 = uniform).
                Defaults to 0.1.

        Returns:
            EdgeEmittingHMM: HMM instance with latent_dim=N and emission_dim=2*width+1.
        """
        num_latent_states, parent_of, which_child_of, is_leaf, is_root, children_arr, degree = _dyck_build_tree(
            depth, width
        )
        emission_dim = 2 * width + 1

        # Transfer matrix: uniform local walk + temperature teleportation.
        T_local = np.zeros((num_latent_states, num_latent_states))
        for j in range(num_latent_states):
            if not is_root[j]:
                T_local[parent_of[j], j] = 1.0 / degree[j]
            if not is_leaf[j]:
                for m in range(width):
                    T_local[children_arr[j, m], j] = 1.0 / degree[j]
        T_matrix = (1.0 - temperature) * T_local + temperature / num_latent_states

        # Emission tensor: E[y, i, j] = P(y | x_t=i, x_{t-1}=j).
        E_tensor = np.zeros((emission_dim, num_latent_states, num_latent_states))
        E_tensor[0] = 1.0
        non_root = np.arange(1, num_latent_states)
        E_tensor[0, non_root, parent_of[1:]] = 0.0
        E_tensor[2 * which_child_of[1:] + 1, non_root, parent_of[1:]] = 1.0
        E_tensor[0, parent_of[1:], non_root] = 0.0
        E_tensor[2 * which_child_of[1:] + 2, parent_of[1:], non_root] = 1.0

        stationary = np.ones(num_latent_states) / num_latent_states if temperature == 0 else None

        return EdgeEmittingHMM(
            latent_dim=num_latent_states,
            emission_dim=emission_dim,
            transfer_operator=T_matrix,
            emission_operator=E_tensor,
            latent_stationary_density=stationary,
        )

    @staticmethod
    def dyck_fun(
        depth: int = 3,
        width: int = 2,
        temperature: float = 0.1,
    ) -> EdgeEmittingHMM:
        """Construct a functional Dyck language HMM on a complete width-ary tree.

        Suitable for large trees where materialising the full transfer matrix or
        emission tensor is prohibitive. Operators are JAX-traceable closures over
        precomputed tree-structure arrays; no N×N matrix is ever stored. For small
        trees use ``dyck_arr``.

        The HMM has N = (width^(depth+1) - 1) / (width - 1) latent states.
        Transfer and emission semantics are identical to ``dyck_arr``.

        Args:
            depth (int): Tree depth. Defaults to 3.
            width (int): Branching factor. Defaults to 2.
            temperature (float): Teleportation probability (0 = pure walk, 1 = uniform).
                Defaults to 0.1.

        Returns:
            EdgeEmittingHMM: HMM instance with latent_dim=N and emission_dim=2*width+1.
        """
        num_latent_states, parent_of, which_child_of, is_leaf, is_root, children_arr, degree = _dyck_build_tree(
            depth, width
        )
        emission_dim = 2 * width + 1

        # JAX arrays captured by the functional closures below.
        parent_jax = jnp.array(parent_of)
        which_child_jax = jnp.array(which_child_of)
        not_root = jnp.array(~is_root, dtype=jnp.float32)
        not_leaf = jnp.array(~is_leaf, dtype=jnp.float32)
        children_jax = jnp.array(children_arr)
        degree_jax = jnp.array(degree)
        degree_at_parent = degree_jax[parent_jax]

        def transfer_func(x: jax.Array) -> jax.Array:
            up = x * not_root / degree_jax
            recv_from_children = jnp.zeros(num_latent_states).at[parent_jax].add(up)
            recv_from_parent = x[parent_jax] * not_root / degree_at_parent
            local_step = recv_from_children + recv_from_parent
            return (1.0 - temperature) * local_step + (temperature / num_latent_states) * jnp.sum(x) * jnp.ones(
                num_latent_states
            )

        def emission_func(x_next: jax.Array, x_prev: jax.Array) -> jax.Array:
            x_next_padded = jnp.concatenate([x_next, jnp.zeros(1)])
            opening = (x_next_padded[children_jax] * x_prev[:, None] * not_leaf[:, None]).sum(axis=0)
            closing_nodes = x_next[parent_jax] * x_prev * not_root
            closing = jnp.zeros(width).at[which_child_jax].add(closing_nodes)
            no_bracket = jnp.sum(x_next) * jnp.sum(x_prev) - opening.sum() - closing.sum()
            open_close = jnp.stack([opening, closing], axis=1).ravel()
            return jnp.concatenate([no_bracket[None], open_close])

        stationary = np.ones(num_latent_states) / num_latent_states if temperature == 0 else None

        return EdgeEmittingHMM(
            latent_dim=num_latent_states,
            emission_dim=emission_dim,
            transfer_operator=transfer_func,
            emission_operator=emission_func,
            latent_stationary_density=stationary,
        )

    @staticmethod
    def dishonest_casino() -> NodeEmittingHMM:
        """Construct the "dishonest casino" HMM.

        A two-state HMM modelling a fair die versus a loaded die (biased toward 6).
        Both states have high self-transition probabilities, producing sticky latent dynamics.

        Returns:
            DenseHMM: HMM instance with latent_dim=2 and emission_dim=6.
        """
        return NodeEmittingHMM(
            latent_dim=2,
            emission_dim=6,
            transfer_operator=np.array([[0.95, 0.10], [0.05, 0.90]]),
            emission_operator=np.array(
                [
                    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                    [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
                ]
            ).T,
        )

    @staticmethod
    def random_dirichlet(
        latent_dim: int = 2,
        emission_dim: int = 2,
        transfer_concentration: float = 0.9,
        emission_concentration: float = 0.9,
    ) -> NodeEmittingHMM:
        """Construct a random HMM with Dirichlet-sampled matrices.

        The transfer and emission matrices are constructed from columns sampled
        independently from Dirichlet distributions.

        Args:
            latent_dim (int, optional): Number of hidden states. Defaults to 2.
            emission_dim (int, optional): Number of emission symbols. Defaults to 2.
            transfer_concentration (float, optional): Dirichlet concentration for the transfer matrix.
                Values < 1 produce sparser, peakier columns; values > 1 produce more uniform columns.
            emission_concentration (float, optional): Dirichlet concentration for the emission matrix.
                Values < 1 produce sparser, peakier columns; values > 1 produce more uniform columns.

        Returns:
            DenseHMM: HMM instance.
        """
        transfer_matrix = np.random.dirichlet(transfer_concentration * np.ones(latent_dim), size=latent_dim).T
        emission_matrix = np.random.dirichlet(emission_concentration * np.ones(emission_dim), size=latent_dim).T
        return NodeEmittingHMM(
            latent_dim=latent_dim,
            emission_dim=emission_dim,
            transfer_operator=transfer_matrix,
            emission_operator=emission_matrix,
        )

    @staticmethod
    def pip(
        dimension: int = 3,
        tau: float = 0.15,
        eps: float = 0.20,
    ) -> NodeEmittingHMM:
        transfer_matrix = np.eye(dimension)*(1-dimension*tau) + np.ones((dimension,dimension))*tau
        emission_matrix = np.eye(dimension)*(1-dimension*eps) + np.ones((dimension,dimension))*eps
        return NodeEmittingHMM(
            latent_dim=dimension,
            emission_dim=dimension,
            transfer_operator=transfer_matrix,
            emission_operator=emission_matrix,
        )