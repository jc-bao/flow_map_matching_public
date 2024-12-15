# flow_map_matching

Minimal implementation of the flow map matching method (https://arxiv.org/abs/2406.07507) in ``jax``.

Note: this is an unofficial implementation and does not recreate the exact experiments of the paper, but does contain implementations of the associated loss functions and basic training loops.

Also contains a modified version of the HuggingFace Diffusers UNet implementation to allow for two times as needed by the flow map formalism, along with a minimal stochastic interpolant implementation.

