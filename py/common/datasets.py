"""
Nicholas M. Boffi
4/15/24

Code for loading in jax datasets using torchvision
"""

import torchvision
import jax.numpy as np
from ml_collections import config_dict


## map image from [-1, 1] to [0, 1]
def unnormalize_image(image: np.ndarray):
    return (image + 1) / 2


## map image from [0, 255] to [-1, 1]
def normalize_image(image: np.ndarray):
    return (2 * (np.float32(image) / 255)) - 1


def get_dataset(cfg: config_dict.ConfigDict):
    """Load in a dataset using torchvision and return it as a dictionary."""

    if cfg.target == "mnist":
        mnist = {
            "train": torchvision.datasets.MNIST(
                "/scratch/nb3397/datasets", train=True, download=True
            ),
            "test": torchvision.datasets.MNIST(
                "/scratch/nb3397/datasets", train=False, download=True
            ),
        }

        ds = {}

        for split in ["train", "test"]:
            ds[split] = {
                "image": mnist[split].data.numpy(),
                "label": mnist[split].targets.numpy(),
            }

            # cast from np to jnp and rescale the pixel values from [0,255] to [-1,1]
            ds[split]["image"] = normalize_image(ds[split]["image"])
            ds[split]["label"] = np.float32(ds[split]["label"])

            # torchvision returns shape (B, 28, 28).
            # hence, append the trailing channel dimension.
            ds[split]["image"] = np.expand_dims(ds[split]["image"], 3)

        return ds["train"], ds["test"]

    elif cfg.target == "cifar10":
        cifar10 = {
            "train": torchvision.datasets.CIFAR10(
                "/scratch/nb3397/datasets", train=True, download=True
            ),
            "test": torchvision.datasets.CIFAR10(
                "/scratch/nb3397/datasets", train=False, download=True
            ),
        }

        ds = {}

        for split in ["train", "test"]:
            ds[split] = {
                "image": np.array(cifar10[split].data),
                "label": np.array(cifar10[split].targets),
            }

            # cast from np to jnp and rescale the pixel values from [0,255] to [-1,1]
            ds[split]["image"] = normalize_image(ds[split]["image"])
            ds[split]["label"] = np.float32(ds[split]["label"])

        return ds["train"], ds["test"]
    else:
        raise ValueError(f"Unknown dataset: {cfg.target}")
