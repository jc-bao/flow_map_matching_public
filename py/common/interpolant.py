import jax
import jax.numpy as np
import dataclasses
import functools
from typing import Callable


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant"""

    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    beta_dot: Callable[[float], float]

    def calc_It(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(self, t: float, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It(t, x0, x1)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It_dot(
        self, t: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return self.calc_It_dot(t, x0, x1)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta
