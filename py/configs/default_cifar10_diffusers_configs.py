"""
Nicholas M. Boffi
4/15/24

Default config for training a flow map matching model on CIFAR-10.
"""

import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()

    ## data
    config.data = ml_collections.ConfigDict()
    config.network_type = "diffusers_unet"
    config.target = "cifar10"

    ## model
    config.map_to_ve = False

    config.diffuser_config = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "only_cross_attention": False,
        "block_out_channels": (128, 256, 256, 256),
        # "block_out_channels": (64, 128),
        "layers_per_block": 2,
        # "layers_per_block": 1,
        "num_attention_heads": 4,
        "cross_attention_dim": 256,
        # "cross_attention_dim": 128,
        "dropout": 0.0,
        "use_linear_projection": True,
        "use_memory_efficient_attention": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0.0,
        "num_classes": 10,
        "split_head_dim": True,
    }

    return config
