"""
Nicholas M. Boffi
4/1/24

Loss functions for learning.
"""

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as np
from typing import Dict
import functools
from . import interpolant as interpolant
from . import flow_map as flow_map

Parameters = Dict[str, Dict]


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return np.linalg.norm(flat_params) / np.sqrt(flat_params.size)


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return np.mean(batched_outputs)

    return wrapper


def eulerian_ct(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    label: np.ndarray,
    s: float,
    t: float,
    dropout_keys: np.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Direct 'Eulerian' consistency training-like loss for flow map matching."""
    rng1 = {"dropout": dropout_keys[0]}
    rng2 = {"dropout": dropout_keys[1]}

    # swap s and t if s > t
    s, t = jax.lax.cond(s > t, lambda _: (t, s), lambda _: (s, t), None)

    Is = interp.calc_It(s, x0, x1)
    Is_dot = interp.calc_It_dot(s, x0, x1)
    ds_Xst = X.apply(params, s, t, Is, label, train=True, method="partial_s", rngs=rng1)

    jvp = jax.lax.stop_gradient(
        jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=True, rngs=rng2),
            (Is,),
            (Is_dot,),
        )[1]
    )

    return np.sum((ds_Xst + jvp) ** 2)


def eulerian(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    label: np.ndarray,
    s: float,
    t: float,
    dropout_keys: np.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Direct 'Eulerian' loss for flow map matching."""
    rng1 = {"dropout": dropout_keys[0]}
    rng2 = {"dropout": dropout_keys[1]}
    rng3 = {"dropout": dropout_keys[2]}

    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Xst_It = X.apply(params, s, t, It, label, train=True, rngs=rng1)
    dt_Xts = X.apply(
        params, t, s, Xst_It, label, train=True, method="partial_s", rngs=rng2
    )
    jvp = jax.jvp(
        lambda x: X.apply(params, s, t, x, label, train=True, rngs=rng3),
        (It,),
        (dt_Xts,),
    )[1]

    return np.sum((jvp + It_dot) ** 2)


def lagrangian(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    label: np.ndarray,
    s: float,
    t: float,
    dropout_keys: np.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Direct 'Lagrangian' loss for flow map matching."""
    rng1 = {"dropout": dropout_keys[0]}
    rng2 = {"dropout": dropout_keys[1]}

    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)
    Xts_It = X.apply(params, t, s, It, label, train=True, rngs=rng1)
    dt_Xst = X.apply(
        params, s, t, Xts_It, label, train=True, method="partial_t", rngs=rng2
    )

    return np.sum((dt_Xst - It_dot) ** 2)


def distill(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    label: np.ndarray,
    s: float,
    t: float,
    dropout_keys: np.ndarray,
    teacher_params: Parameters,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Distillation loss for flow map matching."""
    rng = {"dropout": dropout_keys[0]}

    # swap s and t if s > t
    s, t = jax.lax.cond(s > t, lambda _: (t, s), lambda _: (s, t), None)

    Is = interp.calc_It(s, x0, x1)
    Xst_Is = X.apply(params, s, t, Is, label, train=True, rngs=rng)
    step1 = X.apply(teacher_params, s, 0.5 * (s + t), Is, label, train=False)
    target = X.apply(teacher_params, 0.5 * (s + t), t, step1, label, train=False)

    return np.sum((Xst_Is - target) ** 2)


def lagrangian_distill(
    params: Parameters,
    x0: np.ndarray,
    x1: np.ndarray,
    label: np.ndarray,
    s: float,
    t: float,
    dropout_keys: np.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    apply_b: Callable,
    b_params: Parameters,
) -> float:
    """Lagrangian distillation from a pre-trained flow."""
    rng1 = {"dropout": dropout_keys[0]}
    rng2 = {"dropout": dropout_keys[1]}

    # compute the evaluation point
    Is = interp.calc_It(s, x0, x1)
    Xst_Is = X.apply(params, s, t, Is, label, train=True, rngs=rng1)

    # compute the teacher at the evaluation point
    b_eval = apply_b(b_params, t, Xst_Is, label, train=False)

    # compute the time derivative at the evaluation point
    dt_Xst = X.apply(
        params, s, t, Xst_Is, label, train=True, method="partial_t", rngs=rng2
    )

    # minimize square residual
    return np.sum((dt_Xst - b_eval) ** 2)
