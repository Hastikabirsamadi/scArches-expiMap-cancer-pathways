"""
Clean UMAP figures for the CTCL mapping. Each figure answers one question.

    python plot_ctcl.py --out DRIVE/ctcl_map/seed0

Reads joint_latent.h5ad from --out (written by map_ctcl.py), so no training is needed to redraw.
In every panel all cells keep the same position. Cells that are not the subject of the panel are drawn
in light grey behind, so the coloured cells can be seen against the whole map.

fig1_cell_types.png   did CTCL cells land on the right healthy cell types?
fig2_disease.png      where are healthy, Sezary and MF cells, and where are the malignant cells?
fig3_author_labels.png  the authors' own labels, to compare with fig1
fig4_confidence.png   how sure the label transfer was (map and histogram)
fig5_mixing.png       do healthy CTCL donors mix with the reference? and which study is where?
"""
import argparse
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GREY = "#dddddd"
DISEASE_COLORS = {"normal": "#4c9be8", "Sezary syndrome": "#e8744c", "mycosis fungoides": "#9b59b6"}


def palette(names):
    """Fixed colour per name (tab20, then tab20b if more are needed)."""
    cols = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return {n: cols[i % len(cols)] for i, n in enumerate(sorted(names))}


def base(ax, xy, title):
    ax.scatter(xy[:, 0], xy[:, 1], s=1, c=GREY, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def categories(ax, xy, mask, labels, colors, title, legend=True, order=None, size=2):
    """Grey map of all cells, then the masked cells coloured by label (drawn in random order, so no
    group hides another just because it was drawn last)."""
    base(ax, xy, title)
    idx = np.where(mask)[0]
    idx = idx[np.random.RandomState(0).permutation(len(idx))]
    ax.scatter(xy[idx, 0], xy[idx, 1], s=size, c=[colors[v] for v in labels[idx]], linewidths=0,
               rasterized=True)
    if legend:
        names = order or sorted(set(labels[mask]))
        counts = {n: int((labels[mask] == n).sum()) for n in names}
        handles = [plt.Line2D([], [], marker="o", ls="", color=colors[n], ms=6) for n in names]
        ax.legend(handles, [f"{n} ({counts[n]:,})" for n in names], loc="center left",
                  bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def make_figures(joint, out, disease_key="disease", label_key="author_sub_cell_type", normal="normal"):
    xy = joint.obsm["X_umap"]
    obs = joint.obs
    is_ref = (obs["source"].astype(str) == "reference").values
    is_q = ~is_ref
    ct = obs["cell_type"].astype(str).values
    ct_colors = palette(set(ct))

    # 1. cell types: reference (true labels) next to CTCL (labels voted by the 15 nearest reference cells)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    categories(axes[0], xy, is_ref, ct, ct_colors, "Healthy reference: cell types", legend=False)
    categories(axes[1], xy, is_q, ct, ct_colors, "CTCL cells: cell type voted by nearest reference cells")
    fig.suptitle("Did the CTCL cells land on the right cell types? Same colour = same cell type", y=1.0)
    save(fig, out, "fig1_cell_types.png")

    # 2. disease and malignant cells
    dis = obs[disease_key].astype(str).values
    lab = obs[label_key].astype(str).values
    mal = obs[label_key].astype(str).str.lower().str.contains("malignant").values
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw=dict(wspace=0.55))
    dcol = {**palette(set(dis[is_q])), **DISEASE_COLORS}
    categories(axes[0], xy, is_q, dis, dcol, "CTCL cells by disease")
    group = np.where(mal, "malignant (authors' label)", "other CTCL cells")
    categories(axes[1], xy, is_q, group, {"malignant (authors' label)": "#d62728", "other CTCL cells": "#9ecae1"},
               "Malignant cells", order=["other CTCL cells", "malignant (authors' label)"])
    fig.suptitle("Grey = healthy reference cells", y=1.0)
    save(fig, out, "fig2_disease.png")

    # 3. the authors' own labels (18 groups, so it gets a figure of its own)
    fig, ax = plt.subplots(figsize=(9, 6))
    categories(ax, xy, is_q, lab, palette(set(lab[is_q])), "CTCL cells: the authors' own cell labels")
    save(fig, out, "fig3_author_labels.png")

    # 4. confidence of the label transfer
    conf = obs["pred_confidence"].astype(float).values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.45))
    base(axes[0], xy, "CTCL cells: confidence of the voted cell type")
    idx = np.where(is_q)[0]
    idx = idx[np.argsort(-conf[idx])]  # low confidence drawn last, so it stays visible
    sca_ = axes[0].scatter(xy[idx, 0], xy[idx, 1], s=2, c=conf[idx], cmap="viridis", vmin=0, vmax=1,
                           linewidths=0, rasterized=True)
    cb = fig.colorbar(sca_, ax=axes[0], fraction=0.04)
    cb.set_label("share of the 15 nearest reference cells\nthat agree (1 = all 15)")
    bins = np.linspace(0, 1, 16)
    for d in sorted(set(dis[is_q])):
        m = is_q & (dis == d)
        axes[1].hist(conf[m], bins=bins, histtype="step", lw=2, density=True, color=dcol[d],
                     label=f"{d} (median {np.median(conf[m]):.2f})")
    axes[1].set_xlabel("confidence"), axes[1].set_ylabel("share of cells (density)")
    axes[1].set_title("Confidence per disease group", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig4_confidence.png")

    # 5. mixing: reference vs healthy CTCL donors vs patients, and the study of each cell
    src = np.where(is_ref, "healthy reference", np.where(dis == normal, "CTCL data: healthy donors",
                                                         "CTCL data: patients"))
    scol = {"healthy reference": "#ff7f0e", "CTCL data: healthy donors": "#1f77b4",
            "CTCL data: patients": "#2ca02c"}
    study = obs["study"].astype(str).values
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw=dict(wspace=0.6))
    both = is_ref | (dis == normal)
    categories(axes[0], xy, both, src, scol, "Healthy cells only: reference vs CTCL healthy donors",
               order=["healthy reference", "CTCL data: healthy donors"])
    categories(axes[1], xy, is_q, study, palette(set(study[is_q])), "CTCL cells by study (batch)")
    fig.suptitle("Good mixing = orange and blue dots sit together, not side by side", y=1.0)
    save(fig, out, "fig5_mixing.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder with joint_latent.h5ad; figures are saved here")
    ap.add_argument("--disease_key", default="disease")
    ap.add_argument("--label_key", default="author_sub_cell_type")
    ap.add_argument("--normal_value", default="normal")
    args = ap.parse_args()
    joint = ad.read_h5ad(os.path.join(args.out, "joint_latent.h5ad"))
    make_figures(joint, args.out, args.disease_key, args.label_key, args.normal_value)


if __name__ == "__main__":
    main()
