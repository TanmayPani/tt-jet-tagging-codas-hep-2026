import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import random
    import numpy as np
    from matplotlib import pyplot as plt
    from jet_util import (
        load_images,
        anti_kt_clustering,
        extract_cluster_features,
        image_to_constituents,
        canonicalize_jet,
        pixelize,
    )


@app.function
def plot_jet_sample(ax, idx, X, y, uid, clusters=None, **kwargs):
    ax.imshow(X[idx], norm="log", extent=(-1.5, 1.5, -1.5, 1.5))
    ax.set_title(f"{uid[idx]}, y={y[idx]}")
    ax.set_ylabel(r"$\eta$")
    ax.set_xlabel(r"$\phi$")
    if clusters is not None:
        pass
    for key, val in kwargs.items():
        if key.startswith("scatter__"):
            x, y = val[idx][0], val[idx][1]
            ax.scatter(
                x,
                y,
                color="red",
                label=f"{key.lstrip('scatter__')}, ({x:.1f}, {y:.1f})",
            )
    ax.axvline(0, linestyle="--", color="grey")
    ax.axhline(0, linestyle="--", color="grey")
    # ax.legend()
    return ax


@app.function
def plot_random_jet_samples(shape, X, y, uid, figscale=5, seed=42, **kwargs):
    random.seed(seed)
    fig, axs = plt.subplots(
        *shape, figsize=tuple(d * figscale for d in reversed(shape))
    )
    axs = axs.flatten()
    for ax in axs:
        idx = random.randrange(X.shape[0])

        _ = plot_jet_sample(ax, idx, X, y, uid, **kwargs)

    return fig


@app.cell
def _():
    input_arrays = load_images()
    return (input_arrays,)


@app.cell
def _(input_arrays):
    for _iarr, _arr in enumerate(input_arrays):
        print(f"load_images(): arr# {_iarr}, shape={_arr.shape}, type={_arr.dtype}")
    _X, _y, _ids = input_arrays[0:3]

    _X_scaled = _X / _X.sum(axis=(1, 2), keepdims=True)

    _, _nx, _ny, _ = _X_scaled.shape
    _mgrid = np.meshgrid(np.arange(_nx), np.arange(_ny))
    # print(*(_g for _g in _mgrid))
    _centroid = np.concatenate(
        [
            (np.tensordot(_g, _X_scaled, axes=((0, 1), (1, 2))) - 15) * 0.1
            for _g in _mgrid
        ],
        axis=-1,
    )
    print(_centroid.shape)

    _figs = []
    _figs.append(
        plot_random_jet_samples(
            (2, 5), _X, _y, _ids, seed=100, scatter__centroid=_centroid
        )
    )

    _cent_pos = _centroid[_y > 0.5]
    _cent_neg = _centroid[_y < 0.5]
    _fig_cent, _ax_cent = plt.subplots(1, 1, figsize=(5, 5))
    _ax_cent.set_xlim(-1.5, 1.5)
    _ax_cent.set_ylim(-1.5, 1.5)
    _ax_cent.yaxis.set_inverted(True)
    _ax_cent.scatter(_cent_pos[:, 0], _cent_pos[:, 1], color="red", label="y=1")
    _ax_cent.scatter(_cent_neg[:, 0], _cent_neg[:, 1], color="blue", label="y=0")
    _ax_cent.axvline(0, linestyle="--", color="grey")
    _ax_cent.axhline(0 / 2, linestyle="--", color="grey")
    _ax_cent.legend()

    _figs.append(_fig_cent)

    mo.vstack(_figs)
    # mo.vstack([mo.ui.matplotlib(_f) for _f in _figs])
    return


@app.cell
def _(input_arrays):
    _X, _y, _uid = input_arrays[0:3]
    _clusters = anti_kt_clustering(_X[0])
    _fig, _ax = plt.subplots(1, 1, figsize=(5, 5))

    mo.hstack(
        (
            plot_jet_sample(_ax, 0, _X, _y, _uid),
            mo.json(extract_cluster_features(_clusters)),
            mo.tree(_clusters),
        )
    )
    return


@app.cell
def _(input_arrays):
    # canonicalization on several different jets (mix of QCD and top)
    _rng = np.random.default_rng(7)
    _pos = np.flatnonzero(input_arrays[1] > 0.5)  # top jets
    _neg = np.flatnonzero(input_arrays[1] < 0.5)  # QCD jets
    _indices = np.concatenate(
        [_rng.choice(_neg, 3, replace=False), _rng.choice(_pos, 3, replace=False)]
    )

    _fig, _axs = plt.subplots(2, len(_indices), figsize=(3 * len(_indices), 6))
    for _k, _idx in enumerate(_indices):
        _cons = image_to_constituents(input_arrays[0][_idx], pt_min=0.0)
        _label = int(input_arrays[1][_idx])
        _axs[0, _k].imshow(pixelize(_cons), norm="log", extent=(-1.5, 1.5, -1.5, 1.5))
        _axs[0, _k].set_title(f"jet {input_arrays[2][_idx]}\n(y={_label})")
        _axs[1, _k].imshow(
            pixelize(canonicalize_jet(_cons)), norm="log", extent=(-1.5, 1.5, -1.5, 1.5)
        )
        _axs[1, _k].set_title("canonicalized")
        for _ax in (_axs[0, _k], _axs[1, _k]):
            _ax.set_xlabel(r"$\phi$")
            _ax.set_ylabel(r"$\eta$")
    _axs[0, 0].set_ylabel("RAW\n" + r"$\eta$")
    _axs[1, 0].set_ylabel("CANONICAL\n" + r"$\eta$")
    _fig.suptitle("canonicalize_jet across different jets (QCD | top)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
