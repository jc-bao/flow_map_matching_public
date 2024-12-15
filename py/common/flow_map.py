import jax
import jax.numpy as np
from flax import linen as nn
from ml_collections import config_dict
import functools
from . import network_utils as network_utils
from typing import Dict


class FlowMap(nn.Module):
    """Basic class for a flow map."""

    config: config_dict.ConfigDict

    def setup(self):
        """Set up the flow map."""
        self.network = network_utils.setup_network(self.config)
        self.flow_map = lambda s, t, x, label, train: (1 - (t - s)) * x + (
            t - s
        ) * self.network(s, t, x, label, train)

        self._partial_s = jax.jacfwd(self.flow_map, argnums=0)
        self._partial_t = jax.jacfwd(self.flow_map, argnums=1)

    def __call__(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Apply the flow map."""
        return self.flow_map(s, t, x, label, train)

    def partial_t(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Compute the partial derivative with respect to time."""
        return self._partial_t(s, t, x, label, train)

    def partial_s(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ) -> np.ndarray:
        """Compute the partial derivative with respect to space."""
        return self._partial_s(s, t, x, label, train)


def sample(
    flow_map: FlowMap, params: Dict, x0: np.ndarray, N: int, label: int
) -> np.ndarray:
    """Unconditional sampling."""
    ts = np.linspace(0.0, 1.0, N + 1)

    def step(x, idx):
        return (
            flow_map.apply(params, ts[idx], ts[idx + 1], x, label=label, train=False),
            None,
        )

    final_state, _ = jax.lax.scan(step, x0, np.arange(N))
    return final_state


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
def batch_sample(
    flow_map: FlowMap, params: Dict, x0s: np.ndarray, N: int, label: int
) -> np.ndarray:
    """Batch unconditional sampling."""
    return sample(flow_map, params, x0s, N, label)
