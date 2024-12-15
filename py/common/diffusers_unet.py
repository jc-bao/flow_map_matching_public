"""
Nicholas M. Boffi
4/25/24

Modifications to the HuggingFace Diffuser UNet architecture.
"""

from typing import Tuple, Union, Optional, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.utils import BaseOutput
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from .unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxCrossAttnUpBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
)


@flax.struct.dataclass
class FlaxUNet2DConditionOutput(BaseOutput):
    """
    The output of [`FlaxUNet2DConditionModel`].

    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: jnp.ndarray


@flax_register_to_config
class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Inherent JAX features such as the following are supported:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be one of `UNetMidBlock2DCrossAttn`. If `None`, the mid block layer
            is skipped.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn"
    only_cross_attention: Union[bool, Tuple[bool]] = False
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1
    projection_class_embeddings_input_dim: Optional[int] = None
    num_classes: int = 10

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros(
            (1, 1, self.cross_attention_dim), dtype=jnp.float32
        )

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None
        return self.init(
            rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs
        )["params"]

    def setup(self) -> None:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0],
            flip_sin_to_cos=self.flip_sin_to_cos,
            freq_shift=self.config.freq_shift,
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # s time
        self.s_time_proj = FlaxTimesteps(
            block_out_channels[0],
            flip_sin_to_cos=self.flip_sin_to_cos,
            freq_shift=self.freq_shift,
        )
        self.s_time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # class embedding
        # allow for null token, so num_classes + 1
        self.class_proj = nn.Embed(
            self.num_classes + 1, block_out_channels[0], dtype=self.dtype
        )

        self.class_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        only_cross_attention = self.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(self.down_block_types)

        if isinstance(self.num_attention_heads, int):
            num_attention_heads = (self.num_attention_heads,) * len(
                self.down_block_types
            )

        # transformer layers per block
        transformer_layers_per_block = self.transformer_layers_per_block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                self.down_block_types
            )

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = FlaxCrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    split_head_dim=self.split_head_dim,
                    dtype=self.dtype,
                )
            else:
                down_block = FlaxDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        if self.config.mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = FlaxUNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                dropout=self.dropout,
                num_attention_heads=num_attention_heads[-1],
                transformer_layers_per_block=transformer_layers_per_block[-1],
                use_linear_projection=self.use_linear_projection,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
        elif self.config.mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"Unexpected mid_block_type {self.config.mid_block_type}")

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block)
        )
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock2D":
                up_block = FlaxCrossAttnUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[
                        i
                    ],
                    num_attention_heads=reversed_num_attention_heads[i],
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    split_head_dim=self.split_head_dim,
                    dtype=self.dtype,
                )
            else:
                up_block = FlaxUpBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample: jnp.ndarray,
        s_timesteps: Union[jnp.ndarray, float, int],
        timesteps: Union[jnp.ndarray, float, int],
        class_labels: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[FlaxUNet2DConditionOutput, Tuple[jnp.ndarray]]:
        r"""
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of
                a plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unets.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 1. time
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        print(f"t_emb.shape: {t_emb.shape}")

        s_emb = self.s_time_proj(s_timesteps)
        s_emb = self.s_time_embedding(s_emb)
        print(f"s_emb.shape: {s_emb.shape}")

        t_emb = t_emb + s_emb

        class_embeds = self.class_proj(class_labels)
        class_embeds = self.class_embedding(class_embeds)

        print(f"class_embeds.shape: {class_embeds.shape}")
        t_emb = t_emb + class_embeds

        # 2. pre-process
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)
        print(f"after conv_in, sample.shape: {sample.shape}")

        # 3. down
        down_block_res_samples = (sample,)
        for kk, down_block in enumerate(self.down_blocks):
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(
                    sample, t_emb, encoder_hidden_states, deterministic=not train
                )
            else:
                sample, res_samples = down_block(sample, t_emb, deterministic=not train)
            print(
                f"after down block {kk+1}/{len(self.down_blocks)}, sample.shape: {sample.shape}"
            )
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample, t_emb, encoder_hidden_states, deterministic=not train
            )
        print(f"after mid block sample.shape: {sample.shape}")

        # 5. up
        for kk, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[
                : -(self.layers_per_block + 1)
            ]
            if isinstance(up_block, FlaxCrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
            else:
                sample = up_block(
                    sample,
                    temb=t_emb,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
            print(
                f"after up block {kk+1}/{len(self.down_blocks)}, sample.shape: {sample.shape}"
            )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        print(f"after conv_norm_out, sample.shape: {sample.shape}")
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        print(f"after conv_out, sample.shape: {sample.shape}")
        sample = jnp.transpose(sample, (0, 3, 1, 2))
        print(f"after transpose, sample.shape: {sample.shape}")

        if not return_dict:
            return (sample,)

        return FlaxUNet2DConditionOutput(sample=sample)
