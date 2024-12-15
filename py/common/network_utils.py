"""
Nicholas M. Boffi
4/1/24

Helper routines for neural network definitions.
"""

import jax
import jax.numpy as np
from typing import Callable
import flax.linen as nn
from ml_collections import config_dict
from common import diffusers_unet as diffusers_unet


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable

    @nn.compact
    def __call__(self, x: np.ndarray):
        for _ in range(self.n_hidden):
            x = nn.Dense(self.n_neurons)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class FlowMapMLP(nn.Module):
    """Simple MLP network with square weight pattern, for flow map representation."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable

    @nn.compact
    def __call__(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ):
        del train
        del label

        st = np.array([s, t])
        inp = np.concatenate((st, x))
        return MLP(self.n_hidden, self.n_neurons, self.output_dim, self.act)(inp)


class DiffusersFlowMapUNet(nn.Module):
    """UNet architecture based on the HuggingFace Diffusers implementation.
    Note: assumes that there is no batch dimension, to interface with the rest of the code.
    However, the diffusersimplementation does have a batch dimension.
    To handle this, we pad batch dimensions to the input.
    """

    config: config_dict.ConfigDict

    @nn.compact
    def __call__(
        self, s: float, t: float, x: np.ndarray, label: float = None, train: bool = True
    ):
        # add batch dimensions
        s = np.array([s], dtype=np.float32)
        t = np.array([t], dtype=np.float32)

        if self.config.map_to_ve:
            s = s * 80
            t = t * 80

        label = np.array(
            [label], dtype=np.int32
        )  # shift to handle null token == -1 (and make sure its not 0)
        x = np.transpose(x, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        x = x.reshape((1, *x.shape))

        # remove batch dimension
        sample = diffusers_unet.FlaxUNet2DConditionModel(**self.config.diffuser_config)(
            x, s, t, label, None, train=train
        )["sample"][0]

        # invert the transpose to go back to (H, W, C)
        return np.transpose(sample, (1, 2, 0))


def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if config.network_type == "mlp":
        return FlowMapMLP(
            n_hidden=config.n_hidden,
            n_neurons=config.n_neurons,
            output_dim=config.d,
            act=get_act(config),
        )
    elif config.network_type == "diffusers_unet":
        return DiffusersFlowMapUNet(config=config)
    else:
        raise ValueError(f"Network type {config.network_type} not recognized.")
