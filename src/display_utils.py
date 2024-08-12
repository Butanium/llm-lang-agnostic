import math
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import torch as th
from matplotlib import markers, font_manager
from pathlib import Path
from contextlib import nullcontext
from IPython.display import display
from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer

PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent

markers_list = [None] + list(markers.MarkerStyle.markers.keys())
simsun_path = PATH / "data/SimSun.ttf"
font_manager.fontManager.addfont(str(simsun_path))
simsun = font_manager.FontProperties(fname=str(simsun_path)).get_name()

plt.rcParams.update({"font.size": 16})
plt_params = dict(linewidth=2.7, alpha=0.8, linestyle="-", marker="o")


def plot_ci_plus_heatmap(
    data,
    heat,
    labels,
    color="blue",
    linestyle="-",
    tik_step=10,
    method="gaussian",
    init=True,
    do_colorbar=False,
    shift=0.5,
    nums=[0.99, 0.18, 0.025, 0.6],
    labelpad=10,
    plt_params=plt_params,
):

    fig, (ax, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 10]}, figsize=(5, 3)
    )
    if do_colorbar:
        fig.subplots_adjust(right=0.8)
    plot_ci(
        ax2,
        data,
        labels,
        color=color,
        linestyle=linestyle,
        tik_step=tik_step,
        method=method,
        init=init,
        plt_params=plt_params,
    )

    y = heat.mean(dim=0)
    x = np.arange(y.shape[0]) + 1

    extent = [
        x[0] - (x[1] - x[0]) / 2.0 - shift,
        x[-1] + (x[1] - x[0]) / 2.0 + shift,
        0,
        1,
    ]
    img = ax.imshow(
        y[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14
    )
    ax.set_yticks([])
    # ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label(
            "entropy", rotation=90, labelpad=labelpad
        )  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2


def plot_ci(
    ax,
    data,
    label,
    color="blue",
    tik_step=10,
    init=True,
    plt_params=plt_params,
):
    if init:
        upper = max(round(data.shape[1] / 10) * 10 + 1, data.shape[1] + 1)
        ax.set_xticks(np.arange(0, upper, tik_step))
        for i in range(0, upper, tik_step):
            ax.axvline(i, color="black", linestyle="--", alpha=0.5, linewidth=1)
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    data_ci = {
        "x": np.arange(data.shape[1]),
        "y": mean,
        "y_upper": mean + (1.96 / (data.shape[0] ** 0.5)) * std,
        "y_lower": mean - (1.96 / (data.shape[0] ** 0.5)) * std,
    }

    df = pd.DataFrame(data_ci)
    # Create the line plot with confidence intervals
    ax.plot(df["x"], df["y"], label=label, color=color, **plt_params)
    ax.fill_between(df["x"], df["y_lower"], df["y_upper"], color=color, alpha=0.3)
    if init:
        ax.spines[["right", "top"]].set_visible(False)


def plot_k(
    axes,
    data,
    label,
    k=4,
    color="blue",
    tik_step=10,
    plt_params=plt_params,
    init=True,
    same_scale=True,
):
    if len(axes) < k:
        raise ValueError("Number of axes must be greater or equal to k")

    for i in range(k):
        ax = axes[i]
        if init:
            upper = max(round(data.shape[1] / 10) * 10 + 1, data.shape[1] + 1)
            ax.set_xticks(np.arange(0, upper, tik_step))
            for j in range(0, upper, tik_step):
                ax.axvline(j, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.plot(data[i], label=label, color=color, **plt_params)
        if init:
            ax.spines[["right", "top"]].set_visible(False)
        if same_scale and init:
            ax.set_ylim(0, 1)


def k_subplots(k, size=(5, 4)) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Returns a figure and axes for plotting k examples.
    """
    n_cols = math.ceil(math.sqrt(k))
    n_rows = math.ceil(k / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(size[0] * n_cols, size[1] * n_rows)
    )
    axes = axes.flatten() if k > 1 else [axes]
    for i in range(k, len(axes)):
        axes[i].axis("off")
    return fig, axes


def plot_topk_tokens(
    next_token_probs,
    tokenizer,
    k=4,
    title=None,
    dynamic_size=True,
    dynamic_color_scale=False,
    use_token_ids=False,
    file=None,
):
    """
    Plot the top k tokens for each layer
    :param probs: Probability tensor of shape (num_layers, vocab_size)
    :param k: Number of top tokens to plot
    :param title: Title of the plot
    :param dynamic_size: If True, the size of the plot will be adjusted based on the length of the tokens
    """
    if isinstance(tokenizer, LanguageModel) or isinstance(
        tokenizer, UnifiedTransformer
    ):
        tokenizer = tokenizer.tokenizer
    if next_token_probs.dim() == 1:
        next_token_probs = next_token_probs.unsqueeze(0)
    if next_token_probs.dim() == 2:
        next_token_probs = next_token_probs.unsqueeze(0)
    num_layers = next_token_probs.shape[1]
    max_token_length_sum = 0
    top_token_indices_list = []
    top_probs_list = []
    for probs in next_token_probs:
        top_tokens = th.topk(probs, k=k, dim=-1)
        top_probs = top_tokens.values
        if not use_token_ids:
            top_token_indices = [
                ["'" + tokenizer.convert_ids_to_tokens(t.item()) + "'" for t in l]
                for l in top_tokens.indices
            ]
        else:
            top_token_indices = [[str(t.item()) for t in l] for l in top_tokens.indices]
        top_token_indices_list.append(top_token_indices)
        top_probs_list.append(top_probs)
    for top_token_indices in top_token_indices_list:
        max_token_length_sum += max(
            [len(token) for sublist in top_token_indices for token in sublist]
        )
    has_chinese = any(
        any("\u4e00" <= c <= "\u9fff" for c in token)
        for top_token_indices in top_token_indices_list
        for sublist in top_token_indices
        for token in sublist
    )

    context = (
        mpl.rc_context(rc={"font.sans-serif": [simsun, "Arial"]})
        if has_chinese
        else nullcontext()
    )
    with context:
        if dynamic_size:
            fig, axes = plt.subplots(
                1,
                len(next_token_probs),
                figsize=(max_token_length_sum * k * 0.25, num_layers / 2 + 1),
            )
        else:
            fig, axes = k_subplots(len(next_token_probs), size=(12, 8))
        if len(next_token_probs) == 1:
            axes = [axes]
        for i, (ax, top_probs, top_token_indices) in enumerate(
            zip(axes, top_probs_list, top_token_indices_list)
        ):
            cmap = sns.diverging_palette(255, 0, as_cmap=True)
            sns_kwargs = {}
            if not dynamic_color_scale:
                sns_kwargs.update(dict(vmin=0, vmax=1, cbar=i == len(axes) - 1))
            sns.heatmap(
                top_probs.detach().numpy(),
                annot=top_token_indices,
                fmt="",
                cmap=cmap,
                linewidths=0.5,
                cbar_kws={"label": "Probability"},
                ax=ax,
                **sns_kwargs,
            )
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Layers")
            ax.set_yticks(np.arange(num_layers) + 0.5, range(num_layers))
        if title is None:
            fig.suptitle(f"Top {k} Tokens Heatmap")
        else:
            fig.suptitle(f"Top {k} Tokens Heatmap - {title}")

        plt.tight_layout()
        if file is not None:
            fig.savefig(file, bbox_inches="tight", dpi=300)
        fig.show()
        plt.show()


def plot_results(
    ax,
    target_probs,
    latent_probs,
    target_lang,
    source_baseline=None,
    target_baseline=None,
):
    colors = sns.color_palette("tab10", 1 + len(latent_probs))
    plot_ci(ax, target_probs, label=target_lang, color=colors[0])
    for i, (label, probs) in enumerate(latent_probs.items()):
        plot_ci(ax, probs, label=label, color=colors[i + 1], init=False)
    if source_baseline is not None:
        ax.axhline(
            source_baseline,
            color=colors[1],
            linestyle="--",
            alpha=0.6,
        )
    if target_baseline is not None:
        ax.axhline(
            target_baseline,
            color=colors[0],
            linestyle="-.",
            alpha=0.6,
        )


def plot_k_results(
    axes,
    target_probs,
    latent_probs,
    target_lang,
    k=None,
):
    if k is None:
        k = len(target_probs)
    colors = sns.color_palette("tab10", 1 + len(latent_probs))
    plot_k(axes, target_probs, label=target_lang, color=colors[0], k=k)
    for i, (label, probs) in enumerate(latent_probs.items()):
        plot_k(axes, probs, label=label, color=colors[i + 1], init=False, k=k)


def display_df(df):
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        display(df)


def printc(x, c="r"):
    m1 = {
        "r": "red",
        "g": "green",
        "y": "yellow",
        "w": "white",
        "b": "blue",
        "p": "pink",
        "t": "teal",
        "gr": "gray",
    }
    m2 = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "pink": "\033[95m",
        "teal": "\033[96m",
        "white": "\033[97m",
        "gray": "\033[90m",
    }
    reset_color = "\033[0m"
    print(f"{m2.get(m1.get(c, c), c)}{x}{reset_color}")
