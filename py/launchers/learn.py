"""
Nicholas M. Boffi
4/12/24

Code for systematic learning of flow maps.
"""

import sys

sys.path.append("../../py")

import jax
import jax.numpy as np
import numpy as onp
import dill as pickle
from typing import Tuple, Callable, Dict
from ml_collections import config_dict
from copy import deepcopy
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import functools
from tqdm.auto import tqdm as tqdm
import wandb
from flax.jax_utils import replicate, unreplicate
import optax
import common.losses as losses
import common.updates as updates
import common.flow_map as flow_map
import common.interpolant as interpolant
import common.gmm as gmm
import common.datasets as datasets
import configs.default_cifar10_diffusers_configs as cifar10_diffusers_configs
import configs.default_mnist_diffusers_configs as mnist_diffusers_configs
from typing import Callable, Tuple
import time

# suppress warnings for timeout
# jax.config.update("jax_debug_nans", True)


####### sensible matplotlib defaults #######
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["grid.color"] = "0.8"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["figure.figsize"] = (8, 4)
mpl.rcParams["figure.titlesize"] = 7.5
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.fontsize"] = 7.5
mpl.rcParams["figure.dpi"] = 300
############################################


Parameters = Dict[str, Dict]


def train_loop(
    x1s: np.ndarray,
    labels: np.ndarray,
    prng_key: np.ndarray,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    schedule: optax.GradientTransformation,
    data: dict,
) -> None:
    """Carry out the training loop."""
    start_time = time.time()
    print(f"Entering training loop. Time: {time.time() - start_time}s")
    loss, distill_loss = setup_loss()
    print(f"Loss function set up. Time: {time.time() - start_time}s")
    anneal_schedule = np.linspace(0.05, 1, cfg.anneal_steps)

    # make sure we're on the GPU/TPU
    params = jax.device_put(data["params"], jax.devices(cfg.device_type)[0])
    teacher_params = None

    if cfg.ndevices > 1:
        params = replicate(params)

    ema_params = {ema_fac: deepcopy(data["params"]) for ema_fac in cfg.ema_facs}

    for curr_epoch in tqdm(range(cfg.n_epochs)):
        if cfg.shuffle_dataset:
            perm = jax.random.permutation(prng_key, np.arange(x1s.shape[0]))
            x1s = x1s[perm]
            if labels is not None:
                labels = labels[perm]
            prng_key = jax.random.split(prng_key)[0]

        print(f"Starting epoch {curr_epoch}. Time: {time.time() - start_time}s")
        pbar = tqdm(range(cfg.nbatches))
        for curr_batch in pbar:
            iteration = curr_batch + curr_epoch * cfg.nbatches

            if iteration == 0:
                print(f"Starting batch {curr_batch}. Time: {time.time() - start_time}s")
            loss_fn_args, prng_key = setup_loss_fn_args(
                x1s,
                labels,
                prng_key,
                curr_batch,
                iteration,
                anneal_schedule,
                teacher_params,
            )
            if iteration == 0:
                print(
                    f"Loss function arguments set up. Time: {time.time() - start_time}s"
                )

            if cfg.loss_type == "distill" and iteration > cfg.distill_steps:
                if iteration == cfg.distill_steps + 1:
                    print(f"Switching to distillation loss on iteration {iteration}.")
                curr_loss = distill_loss
            else:
                curr_loss = loss

            params, opt_state, loss_value, grads = update_fn(
                params, opt_state, opt, curr_loss, loss_fn_args
            )

            if iteration == 0:
                print(f"Parameters updated. Time: {time.time() - start_time}s")

            ## compute EMA params
            if cfg.ndevices > 1:
                curr_params = unreplicate(params)
                loss_value = loss_value[0]
                print(f"loss value: {loss_value}")
            else:
                curr_params = params

            if iteration == 0:
                print(f"Computing EMA parameters. Time: {time.time() - start_time}s")
            ema_params = updates.update_ema_params(curr_params, ema_params, cfg)
            if iteration == 0:
                print(f"EMA parameters computed. Time: {time.time() - start_time}s")

            # update teacher parameters
            if cfg.loss_type == "distill" and ((iteration % cfg.distill_steps) == 0):
                print(f"Updating teacher parameters on iteration {iteration}.")
                teacher_params = deepcopy(params)

            ## log to wandb
            if iteration == 0:
                print(f"Logging metrics. Time: {time.time() - start_time}s")
            data, prng_key = log_metrics(
                data,
                x1s,
                labels,
                iteration,
                curr_params,
                ema_params,
                grads,
                schedule,
                anneal_schedule,
                loss_value,
                loss_fn_args,
                prng_key,
            )
            if iteration == 0:
                print(f"Metrics logged. Time: {time.time() - start_time}s")

            pbar.set_postfix(loss=loss_value)

    # dump one final time
    pickle.dump(data, open(f"{cfg.output_folder}/{cfg.output_name}_final.npy", "wb"))


def make_gmm_plot(
    x1s: np.ndarray,
    params: Parameters,
    prng_key: np.ndarray,
) -> None:
    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = [1, 5, 10, 25]
    titles = ["base and target"] + [rf"${step}$-step" for step in steps]

    ## extract target samples
    plot_x1s = x1s[: cfg.plot_bs]

    ## draw multi-step samples from the model
    x0s = cfg.sample_rho0(cfg.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = onp.zeros((len(steps), cfg.plot_bs, cfg.d))
    for kk, step in enumerate(steps):
        xhats[kk] = flow_map.batch_sample(net, params, x0s, step, -np.ones(cfg.plot_bs))

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax in axs.ravel():
        ax.set_xlim([-7.5, 7.5])
        ax.set_ylim([-7.5, 7.5])
        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0s[:, 0], x0s[:, 1], s=0.1, alpha=0.5, marker="o", c="black")
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )
        else:
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )

            ax.scatter(
                xhats[jj - 1, :, 0],
                xhats[jj - 1, :, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )

    wandb.log({"samples": wandb.Image(fig)})
    return prng_key


def make_projection_plot(
    x1s: np.ndarray,
    params: Parameters,
    prng_key: np.ndarray,
) -> None:
    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = [1, 5, 10, 25]
    titles = [rf"${step}$-step" for step in steps]

    ## extract target samples
    plot_x1s = x1s[: cfg.plot_bs]

    ## draw multi-step samples from the model
    x0s = cfg.sample_rho0(cfg.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = onp.zeros((len(steps), cfg.plot_bs, cfg.d))
    for kk, step in enumerate(steps):
        xhats[kk] = flow_map.batch_sample(net, params, x0s, step, -np.ones(cfg.plot_bs))

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax in axs.ravel():
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)
        sns.kdeplot(plot_x1s[:, 1], ax=ax, fill=True, color="C0", alpha=0.5)
        sns.kdeplot(xhats[jj, :, 1], ax=ax, fill=True, color="C1", alpha=0.5)

    wandb.log({"samples": wandb.Image(fig)})
    return prng_key


def make_image_plot(
    image_dims: Tuple,
    dataset_labels: np.ndarray,
    params: Parameters,
    prng_key: np.ndarray,
) -> None:
    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 1, 1
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 4, 8, 16]
    titles = [rf"{step}-step" for step in steps]

    ## draw multi-step samples from the model
    n_images = min(16, cfg.bs)
    x0s = cfg.sample_rho0(n_images, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = onp.zeros((len(steps), n_images, *image_dims))

    if cfg.conditional:
        if cfg.overfit:
            labels = dataset_labels[:n_images]
        else:
            if cfg.class_dropout > 0:
                labels = np.array(onp.random.choice(cfg.num_classes + 1, n_images))
            else:
                labels = np.array(onp.random.choice(cfg.num_classes, n_images))
    else:
        labels = np.ones(n_images) * cfg.num_classes

    for kk, step in enumerate(steps):
        xhats[kk] = flow_map.batch_sample(net, params, x0s, step, labels)

    ## make the image grids
    nrows = 2 if n_images > 8 else 1
    ncols = n_images // nrows

    for kk, title in enumerate(titles):
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fw * ncols, fh * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = axs.reshape((nrows, ncols))

        fig.suptitle(title, fontsize=fontsize)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")

        ## visualize the generated images
        for ii in range(nrows):
            for jj in range(ncols):
                index = ii * ncols + jj
                image = np.clip(xhats[kk, index], -1, 1)
                image = datasets.unnormalize_image(image)

                if cfg.target == "mnist":
                    axs[ii, jj].imshow(image, cmap="gray")
                else:
                    axs[ii, jj].imshow(image)

        wandb.log({titles[kk]: wandb.Image(fig)})

    # if cfg.overfit, make an identical plot wth the real images
    if cfg.overfit:
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fw * ncols, fh * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = axs.reshape((nrows, ncols))

        fig.suptitle("real images", fontsize=fontsize)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")

        ## visualize the generated images
        for ii in range(nrows):
            for jj in range(ncols):
                index = ii * ncols + jj
                image = np.clip(x1s[index], -1, 1)
                image = datasets.unnormalize_image(image)

                if cfg.target == "mnist":
                    axs[ii, jj].imshow(image, cmap="gray")
                else:
                    axs[ii, jj].imshow(image)

        wandb.log({"real images": wandb.Image(fig)})

    return prng_key


def make_loss_fn_args_plot(
    loss_fn_args: Tuple,
) -> None:
    """Make a plot of the loss function arguments."""
    x0batch, x1batch, _, sbatch, tbatch, _ = loss_fn_args

    # remove pmap reshaping
    x0batch = np.squeeze(x0batch)
    x1batch = np.squeeze(x1batch)
    sbatch = np.squeeze(sbatch)
    tbatch = np.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = interp.batch_calc_It(tbatch, x0batch, x1batch)

    ## set up plot array
    titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$(s, t)$"]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
        else:
            ax.set_xlim([-7.5, 7.5])
            ax.set_ylim([-7.5, 7.5])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 1:
            ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 2:
            ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
        elif jj == 3:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})
    return prng_key


@jax.jit
def setup_loss_fn_args_randomness(
    prng_key: np.ndarray, curr_iter: int, anneal_schedule: np.ndarray
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    # generate needed keys
    tkey, skey, x0key = jax.random.split(prng_key, num=3)

    if (
        cfg.loss_type == "lagrangian"
        or cfg.loss_type == "eulerian_ct"
        or cfg.loss_type == "distill"
    ):
        dropout_keys = jax.random.split(tkey, num=2 * cfg.bs).reshape((cfg.bs, 2, -1))
    elif cfg.loss_type == "eulerian":
        dropout_keys = jax.random.split(tkey, num=3 * cfg.bs).reshape((cfg.bs, 3, -1))
    else:
        raise ValueError("Specified loss is not implemented.")

    new_key = jax.random.split(dropout_keys[0, 0])[0]
    x0batch = cfg.sample_rho0(cfg.bs, x0key)

    if cfg.diagonal_anneal:
        # construct the \delta such that |t-s| < \delta
        maxval = jax.lax.cond(
            curr_iter < cfg.anneal_steps,
            lambda x: anneal_schedule[x],
            lambda _: cfg.tmax,
            operand=curr_iter,
        )
        minval = -maxval

        # draw t randomly over the whole range
        tbatch = jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=cfg.tmin, maxval=cfg.tmax
        )

        # draw s randomly over the range [t - \delta, t + \delta]
        # making sure to clip back to [tmin, tmax]
        sbatch = tbatch + jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=minval, maxval=maxval
        )
        sbatch = np.clip(sbatch, cfg.tmin, cfg.tmax)

    elif cfg.loss_type == "distill":
        k = curr_iter // cfg.distill_steps
        delta = 2**k * cfg.distill_delta

        # draw t randomly over the whole range
        tbatch = jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=cfg.tmin, maxval=cfg.tmax
        )

        # draw s randomly over the range [t - \delta, t + \delta]
        # making sure to clip back to [tmin, tmax]
        sbatch = tbatch + jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=-delta, maxval=delta
        )
        sbatch = np.clip(sbatch, cfg.tmin, cfg.tmax)

    elif cfg.box_anneal:
        # construct the maximum value of both s and t
        maxval = jax.lax.cond(
            curr_iter < cfg.anneal_steps,
            lambda x: anneal_schedule[x],
            lambda _: cfg.tmax,
            operand=curr_iter,
        )

        tbatch = jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=cfg.tmin, maxval=maxval
        )
        sbatch = jax.random.uniform(
            skey, shape=(cfg.bs,), minval=cfg.tmin, maxval=maxval
        )

    else:
        tbatch = jax.random.uniform(
            tkey, shape=(cfg.bs,), minval=cfg.tmin, maxval=cfg.tmax
        )
        sbatch = jax.random.uniform(
            skey, shape=(cfg.bs,), minval=cfg.tmin, maxval=cfg.tmax
        )

    return tbatch, sbatch, x0batch, dropout_keys, new_key


def setup_loss_fn_args(
    x1s: np.ndarray,  # [n, ...]
    labels: np.ndarray,
    prng_key: np.ndarray,
    curr_batch: int,
    curr_iter: int,
    anneal_schedule: np.ndarray,
    teacher_params: Parameters,
) -> Tuple:
    # sort out the indices
    lb = cfg.bs * curr_batch
    ub = lb + cfg.bs

    # set up the randomness
    tbatch, sbatch, x0batch, dropout_keys, new_key = setup_loss_fn_args_randomness(
        prng_key, curr_iter, anneal_schedule
    )
    x1batch = x1s[lb:ub]

    if labels is not None:
        label_batch = labels[lb:ub]
    else:
        label_batch = None
    curr_bs = x1batch.shape[0]

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    if cfg.class_dropout > 0:
        mask = jax.random.bernoulli(new_key, cfg.class_dropout, shape=(curr_bs,))
        mask = mask > 0
        label_batch = label_batch.at[mask].set(cfg.num_classes)
        new_key = jax.random.split(new_key)[0]

    # handle case where batch size does not divide dataset size evenly
    if curr_bs < cfg.bs:
        tbatch = tbatch[:curr_bs]
        sbatch = sbatch[:curr_bs]
        x0batch = x0batch[:curr_bs]
        dropout_keys = dropout_keys[:curr_bs]

    # set up formatting for pmap
    if cfg.ndevices > 1:
        x0batch = x0batch.reshape((cfg.ndevices, -1, *x0batch.shape[1:]))
        x1batch = x1batch.reshape((cfg.ndevices, -1, *x1batch.shape[1:]))
        label_batch = label_batch.reshape((cfg.ndevices, -1))
        sbatch = sbatch.reshape((cfg.ndevices, -1))
        tbatch = tbatch.reshape((cfg.ndevices, -1))
        dropout_keys = dropout_keys.reshape((cfg.ndevices, -1, *dropout_keys.shape[1:]))

    # print shapes of everything
    # print(f"Shapes of loss function arguments:")
    # print(f"x0batch: {x0batch.shape}")
    # print(f"x1batch: {x1batch.shape}")
    # print(f"label_batch: {label_batch.shape}")
    # print(f"sbatch: {sbatch.shape}")
    # print(f"tbatch: {tbatch.shape}")
    # print(f"dropout_keys: {dropout_keys.shape}")

    # set up the loss function arguments
    loss_fn_args = (x0batch, x1batch, label_batch, sbatch, tbatch, dropout_keys)

    if cfg.loss_type == "distill" and curr_iter > cfg.distill_steps:
        loss_fn_args = loss_fn_args + (teacher_params,)

    return loss_fn_args, new_key


def log_metrics(
    data: dict,
    x1s: np.ndarray,
    labels: np.ndarray,
    iteration: int,
    curr_params: Parameters,
    ema_params: Dict[float, Parameters],
    grads: Parameters,
    schedule: optax.GradientTransformation,
    anneal_schedule: np.ndarray,
    loss_value: float,
    loss_fn_args: Tuple,
    prng_key: np.ndarray,
) -> None:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    if cfg.ndevices > 1:
        grads = unreplicate(grads)

    wandb.log(
        {
            f"loss": loss_value,
            f"grad": losses.compute_grad_norm(grads),
            f"learning_rate": schedule(iteration),
            f"anneal_schedule": (
                anneal_schedule[iteration] if iteration < cfg.anneal_steps else 1.0
            ),
        }
    )

    if (iteration % cfg.visual_freq) == 0:
        if "gmm" in cfg.target:
            prng_key = make_gmm_plot(x1s, curr_params, prng_key)
            make_loss_fn_args_plot(loss_fn_args)
        elif cfg.target == "projection":
            prng_key = make_projection_plot(x1s, curr_params, prng_key)
        else:
            start_time = time.time()
            prng_key = make_image_plot(x1s[0].shape, labels, curr_params, prng_key)
            end_time = time.time()
            print(f"Time to make image plot: {end_time - start_time}s")

    if (iteration % cfg.save_freq) == 0:
        data["params"] = jax.device_put(curr_params, jax.devices("cpu")[0])
        data["ema_params"] = jax.device_put(ema_params, jax.devices("cpu")[0])
        pickle.dump(
            data,
            open(
                f"{cfg.output_folder}/{cfg.output_name}_{iteration // cfg.save_freq}.npy",
                "wb",
            ),
        )

    return data, prng_key


def setup_loss() -> Callable:
    if cfg.loss_type == "lagrangian" or cfg.loss_type == "distill":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
        def loss(params, x0, x1, label, s, t, dropout_keys):
            return losses.lagrangian(
                params,
                x0,
                x1,
                label,
                s,
                t,
                dropout_keys,
                interp=interp,
                X=net,
            )

    elif cfg.loss_type == "eulerian":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
        def loss(params, x0, x1, label, s, t, dropout_keys):
            return losses.eulerian(
                params,
                x0,
                x1,
                label,
                s,
                t,
                dropout_keys,
                interp=interp,
                X=net,
            )

    elif cfg.loss_type == "eulerian_ct":

        @losses.mean_reduce
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
        def loss(params, x0, x1, label, s, t, dropout_keys):
            return losses.eulerian_ct(
                params,
                x0,
                x1,
                label,
                s,
                t,
                dropout_keys,
                interp=interp,
                X=net,
            )

    else:
        raise ValueError("Specified loss is not implemented.")

    # always define distillation
    @losses.mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, None))
    def distill_loss(params, x0, x1, label, s, t, dropout_keys, teacher_params):
        return losses.distill(
            params,
            x0,
            x1,
            label,
            s,
            t,
            dropout_keys,
            teacher_params,
            interp=interp,
            X=net,
        )

    return loss, distill_loss


def setup_base(
    cfg: config_dict.ConfigDict, ex_input: np.ndarray
) -> config_dict.ConfigDict:
    """Set up the base density for the system."""
    if cfg.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: np.ndarray):
            return cfg.gaussian_scale * jax.random.normal(
                key, shape=(bs, *ex_input.shape)
            )

        cfg.sample_rho0 = sample_rho0

    elif cfg.base == "sqrt_d_gaussian":
        d = ex_input.size

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: np.ndarray):
            return np.sqrt(d) * jax.random.normal(key, shape=(bs, *ex_input.shape))

        cfg.sample_rho0 = sample_rho0

    else:
        raise ValueError("Specified base density is not implemented.")

    return cfg


def setup_target(cfg: config_dict.ConfigDict, prng_key: np.ndarray) -> np.ndarray:
    """Set up the target density for the system."""
    if "gmm" in cfg.target:
        weights, means, covs = gmm.setup_gmm(cfg.target, cfg.d)
        sample_rho1 = functools.partial(
            gmm.sample_gmm, weights=weights, means=means, covariances=covs
        )
        cfg.sample_rho1 = sample_rho1
        n_samples = cfg.n

        keys = jax.random.split(prng_key, num=(n_samples + 1))
        x1s = cfg.sample_rho1(n_samples, keys)
        labels = None
        prng_key = jax.random.split(keys[-1])[0]

    elif cfg.target == "projection":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho1(
            num_samples: int, key: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            x1s = np.zeros((num_samples, cfg.d))
            gaussians = 1 + 0.25 * jax.random.normal(key, shape=(num_samples,))
            x1s = x1s.at[:, 1].set(gaussians)
            return x1s

        cfg.sample_rho1 = sample_rho1
        n_samples = cfg.n
        key, prng_key = jax.random.split(prng_key)
        x1s = cfg.sample_rho1(n_samples, key)
        labels = None

    elif cfg.target == "mnist" or cfg.target == "cifar10":
        data, _ = datasets.get_dataset(cfg)
        x1s, labels = data["image"], data["label"]
        cfg.num_classes = int(np.max(labels)) + 1

        if cfg.overfit:
            print("Overfitting to a small dataset.")
            inds = onp.random.choice(x1s.shape[0], cfg.n, replace=False)
            x1s = x1s[inds]
            labels = labels[inds]

        print("Loaded image dataset.")
        cfg.n = x1s.shape[0]
        cfg.nbatches = cfg.n // cfg.bs
        cfg.d = int(np.prod(np.array([x1s.shape[1:]])))

        print("Updated config dict with dataset info.")
        print(f"Number of samples: {cfg.n}")
        print(f"Number of batches: {cfg.nbatches}")
        print(f"Dimension of samples: {cfg.d}")
        print(f"Shape of samples: {x1s.shape}")
    else:
        raise ValueError("Specified target density is not implemented.")

    return cfg, x1s, labels, prng_key


def initialize_network(
    ex_input: np.ndarray, prng_key: np.ndarray
) -> Tuple[Parameters, np.ndarray]:
    ex_s = ex_t = 0.0
    ex_label = 0

    params = {
        "params": net.init(prng_key, ex_s, ex_t, ex_input, ex_label, train=False)[
            "params"
        ]
    }
    prng_key = jax.random.split(prng_key)[0]

    print(f"Number of parameters: {jax.flatten_util.ravel_pytree(params)[0].size}")
    return params, prng_key


def parse_command_line_arguments():
    """Process command line arguments and set up associated simulation parameters."""
    parser = argparse.ArgumentParser(description="Elliptic learning.")
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--plot_bs", type=int)
    parser.add_argument("--visual_freq", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--shuffle_dataset", type=int)
    parser.add_argument("--overfit", type=int)
    parser.add_argument("--conditional", type=int)
    parser.add_argument("--class_dropout", type=float)
    parser.add_argument("--box_anneal", type=int)
    parser.add_argument("--diagonal_anneal", type=int)
    parser.add_argument("--anneal_steps", type=int)
    parser.add_argument("--distill_steps", type=int)
    parser.add_argument("--distill_delta", type=float)
    parser.add_argument("--n", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--n_neurons", type=int)
    parser.add_argument("--n_hidden", type=int)
    parser.add_argument("--act", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--decay_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--base", type=str)
    parser.add_argument("--gaussian_scale", type=float)
    parser.add_argument("--target", type=str)
    parser.add_argument("--device_type", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--slurm_id", type=int)

    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()

    if args.target == "cifar10":
        cfg = cifar10_diffusers_configs.get_default_configs()
    elif args.target == "mnist":
        cfg = mnist_diffusers_configs.get_default_configs()
    else:
        cfg = config_dict.ConfigDict()

    cfg.clip = 1.0
    cfg.n_epochs = args.n_epochs
    cfg.bs = args.bs
    cfg.plot_bs = args.plot_bs
    cfg.visual_freq = args.visual_freq
    cfg.save_freq = args.save_freq
    cfg.shuffle_dataset = args.shuffle_dataset
    cfg.overfit = args.overfit
    cfg.conditional = args.conditional
    cfg.class_dropout = args.class_dropout
    cfg.box_anneal = args.box_anneal
    cfg.diagonal_anneal = args.diagonal_anneal
    cfg.anneal_steps = args.anneal_steps
    cfg.distill_steps = args.distill_steps
    cfg.distill_delta = args.distill_delta

    if (cfg.box_anneal + cfg.diagonal_anneal) > 1:
        raise ValueError("Can't do both box annealing and diagonal annealing.")

    cfg.n = args.n
    cfg.nbatches = cfg.n // cfg.bs
    cfg.d = args.d
    cfg.tmin = args.tmin
    cfg.tmax = args.tmax
    cfg.n_neurons = args.n_neurons
    cfg.n_hidden = args.n_hidden
    cfg.act = args.act
    cfg.learning_rate = args.learning_rate
    cfg.decay_steps = args.decay_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.loss_type = args.loss_type
    cfg.base = args.base
    cfg.gaussian_scale = args.gaussian_scale
    cfg.target = args.target
    cfg.device_type = args.device_type
    cfg.wandb_name = f"{args.wandb_name}_{args.slurm_id}"
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.output_folder = args.output_folder
    cfg.output_name = f"{args.output_name}_{args.slurm_id}"
    cfg.slurm_id = args.slurm_id
    cfg.ema_facs = [0.9999]
    cfg.ndevices = jax.local_device_count()

    return cfg


if __name__ == "__main__":
    print("Entering main. Setting up config dict and PRNG key.")
    prng_key = jax.random.PRNGKey(42)
    cfg = setup_config_dict()
    print("Config dict set up. Setting up target.")
    cfg, x1s, labels, prng_key = setup_target(cfg, prng_key)
    print("Target set up. Setting up base.")
    cfg = setup_base(cfg, x1s[0])
    print("Target set up. Freezing config dict.")
    cfg = config_dict.FrozenConfigDict(cfg)  # freeze the config

    ## define and initialize the neural network
    print("Setting up network and initializing.")
    net = flow_map.FlowMap(cfg)
    params, prng_key = initialize_network(x1s[0], prng_key)

    ## define the interpolant
    interp = interpolant.Interpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        alpha_dot=lambda _: -1.0,
        beta_dot=lambda _: 1.0,
    )

    ## define optimizer
    print("Setting up optimizer.")
    if cfg.decay_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=int(cfg.warmup_steps),
            decay_steps=int(cfg.decay_steps),
        )

        opt = optax.chain(
            optax.clip_by_global_norm(cfg.clip), optax.radam(learning_rate=schedule)
        )

    else:
        # construct a constant schedule
        schedule = optax.constant_schedule(cfg.learning_rate)

        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.radam(learning_rate=cfg.learning_rate),
        )

    ## set up update function
    update_fn = updates.pupdate if cfg.ndevices > 1 else updates.update
    print(f"update_fn is {update_fn}")

    # for parallel training
    opt_state = opt.init(params)
    if cfg.ndevices > 1:
        opt_state = replicate(opt_state)

    print("Optimizer set up.")

    ## set up weights and biases tracking
    print("Setting up wandb.")
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        config=cfg.to_dict(),
        mode="online",
    )
    print("Wandb set up.")
    print("nbatches:", cfg.nbatches)

    ## train the model
    data = {
        "params": params,
        "cfg": cfg,
    }

    train_loop(x1s, labels, prng_key, opt, opt_state, schedule, data)
