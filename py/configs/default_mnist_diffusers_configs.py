"""
Nicholas M. Boffi
5/16/24
"""

import ml_collections


def get_default_configs():
    config = ml_collections.ConfigDict()

    ## data
    config.data = ml_collections.ConfigDict()
    config.network_type = "diffusers_unet"
    config.target = "mnist"

    ## model
    config.map_to_ve = False

    config.diffuser_config = {
        "sample_size": 28,
        "in_channels": 1,
        "out_channels": 1,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            #            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            #            "CrossAttnUpBlock2D",
        ),
        "only_cross_attention": False,
        # "block_out_channels": (64, 128, 128),
        "block_out_channels": (64, 128),
        "layers_per_block": 2,
        "num_attention_heads": 4,
        "cross_attention_dim": 128,
        "dropout": 0.0,
        "use_linear_projection": True,
        "use_memory_efficient_attention": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0.0,
        "num_classes": 10,
        "split_head_dim": True,
    }

    return config
