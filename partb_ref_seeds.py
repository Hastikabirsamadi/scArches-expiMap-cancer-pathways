"""
Part B0: are the pathway scores already different between seeds on HEALTHY reference cells?

    python partb_ref_seeds.py --refs RUNS/seed0/ref_model RUNS/seed1/ref_model RUNS/seed2/ref_model --out OUT

No cancer data and no training: the three reference models of the baseline project (seeds 0, 1, 2)
score the same healthy PBMC cells. For every pathway active in all seeds:
1. cell level:   do two seeds give the same cell the same score? (Pearson r over all cells)
2. group level:  do two seeds agree on how much a pathway differs between two cell types?
                 (a large contrast, CD4+ T vs CD14+ monocytes, and a small one, CD4+ T vs CD8+ T;
                 Spearman r over pathways, like the CTCL seed check)
3. decoder:      do two seeds use the same genes with the same weights for a pathway? (cosine similarity
                 of the decoder weight vectors)
If the seeds already disagree here, the instability comes from training the reference, not from surgery
or from the cancer data.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
warnings.simplefilter("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scarches as sca  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from map_ctcl import latent  # noqa: E402
from pathways_ctcl import pathway_scores  # noqa: E402
from run_experiment import prepare  # noqa: E402


def contrast(Z, ct, a, b):
    """Per pathway: mean score of cell type a minus cell type b (scores are in SD)."""
    return Z[ct == a].mean(0) - Z[ct == b].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, help="reference model folders, one per seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--contrasts", default="CD4+ T cells:CD14+ Monocytes,CD4+ T cells:CD8+ T cells")
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    adata, _ = prepare(args.data, args.ref_max_cells, 0)
    ct = adata.obs["cell_type"].astype(str).values
    scores, weights, seeds = {}, {}, []
    for path in args.refs:
        name = os.path.basename(os.path.dirname(os.path.normpath(path))) or path
        saved = np.genfromtxt(os.path.join(path, "var_names.csv"), delimiter=",", dtype=str)
        a = adata[:, list(saved)].copy()
        model = sca.models.EXPIMAP.load(path, a)
        z = latent(model, a.X, a.obs["study"])
        s, names = pathway_scores(model, z, a.uns["terms"])
        Z = (s - s.mean(0)) / (s.std(0) + 1e-8)
        scores[name] = pd.DataFrame(Z, columns=names)
        # decoder weights of the active pathways, signed like the scores, genes in one fixed order
        W = model.model.decoder.L0.expr_L.weight.data.cpu().numpy()[:, model.nonzero_terms()]
        sign = np.sign(W.sum(0))
        sign[sign == 0] = 1
        weights[name] = pd.DataFrame(W * sign, index=saved, columns=names).reindex(adata.var_names).fillna(0)
        seeds.append(name)
        print(name, "active pathways:", len(names), flush=True)

    common = sorted(set.intersection(*[set(scores[s].columns) for s in seeds]))
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    res = {"seeds": seeds, "active_per_seed": {s: int(scores[s].shape[1]) for s in seeds},
           "active_in_all": len(common)}
    rows = []
    for a, b in pairs:
        A, B = scores[a][common].values, scores[b][common].values
        r_cell = np.array([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(len(common))])
        wa, wb = weights[a][common].values, weights[b][common].values
        wa, wb = wa.astype(np.float64), wb.astype(np.float64)
        na, nb = np.linalg.norm(wa, axis=0), np.linalg.norm(wb, axis=0)
        # no fixed epsilon here: an earlier version added 1e-12, which turned the cosine of pathways with
        # tiny (almost switched off) weights into 0. Active pathways never have a norm of exactly 0.
        cos = (wa * wb).sum(0) / (na * nb)
        for j, n in enumerate(common):
            rows.append(dict(pair=f"{a}_vs_{b}", pathway=n, cell_r=r_cell[j], weight_cosine=cos[j],
                             weight_norm_a=na[j], weight_norm_b=nb[j]))
        res[f"{a}_vs_{b}"] = {"median_cell_r": round(float(np.median(r_cell)), 3),
                              "share_cell_r_above_0.5": round(float((r_cell > 0.5).mean()), 3),
                              # strongly negative r = same pathway, but the sign rule flipped it in one seed
                              "share_cell_r_below_-0.5": round(float((r_cell < -0.5).mean()), 3),
                              "median_abs_cell_r": round(float(np.median(np.abs(r_cell))), 3),
                              "median_weight_cosine": round(float(np.median(cos)), 3)}
        for c in args.contrasts.split(","):
            x, y = c.split(":")
            if (ct == x).any() and (ct == y).any():
                r = spearmanr(contrast(A, ct, x, y), contrast(B, ct, x, y))[0]
                res[f"{a}_vs_{b}"][f"contrast_spearman[{x} vs {y}]"] = round(float(r), 3)
    # "almost off": the pathway's weights are below 1% of the seed's median weight size. Group lasso shrinks
    # unused pathways towards zero, and some stop just short of exactly zero, so they still count as active
    # although the decoder hardly uses them. Their scores are then mostly noise.
    norms = pd.DataFrame({s: np.linalg.norm(weights[s][common].values.astype(np.float64), axis=0) for s in seeds},
                         index=common)
    small = norms < 0.01 * norms.median()
    norms.assign(**{f"almost_off_{s}": small[s] for s in seeds}).to_csv(os.path.join(args.out, "weight_norms.csv"))
    res["almost_off_per_seed"] = {s: int(small[s].sum()) for s in seeds}
    clear = [n for n in common if not small.loc[n].any()]
    res["clearly_active_in_all"] = len(clear)
    per = pd.DataFrame(rows)
    for a, b in pairs:
        g = per[(per["pair"] == f"{a}_vs_{b}") & per["pathway"].isin(clear)]
        res[f"{a}_vs_{b}"]["clearly_active_median_cell_r"] = round(float(g["cell_r"].median()), 3)
        res[f"{a}_vs_{b}"]["clearly_active_median_weight_cosine"] = round(float(g["weight_cosine"].median()), 3)
        for c in args.contrasts.split(","):
            x, y = c.split(":")
            if (ct == x).any() and (ct == y).any():
                A, B = scores[a][clear].values, scores[b][clear].values
                r = spearmanr(contrast(A, ct, x, y), contrast(B, ct, x, y))[0]
                res[f"{a}_vs_{b}"][f"clearly_active_contrast_spearman[{x} vs {y}]"] = round(float(r), 3)
    per.to_csv(os.path.join(args.out, "per_pathway.csv"), index=False)
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))

    # figure 1: per-pathway agreement between seeds (cell level and decoder weights)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(wspace=0.3))
    for p, g in per.groupby("pair"):
        axes[0].hist(g["cell_r"], bins=np.linspace(-1, 1, 41), histtype="step", lw=1.5, label=p)
        axes[1].hist(g["weight_cosine"], bins=np.linspace(-1, 1, 41), histtype="step", lw=1.5, label=p)
    axes[0].set_xlabel("same pathway, same cells: score correlation")
    axes[1].set_xlabel("same pathway: gene weight similarity (cosine)")
    for ax in axes:
        ax.set_ylabel("pathways")
        ax.axvline(0, color="grey", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Healthy reference cells, three seeds: 1 = same result, 0 = unrelated", y=1.03)
    fig.savefig(os.path.join(args.out, "fig_B0_agreement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # figure 2: contrast effects, seed 0 against the other seeds (like the CTCL seed check)
    cs = [c.split(":") for c in args.contrasts.split(",")]
    a = seeds[0]
    fig, axes = plt.subplots(len(cs), len(seeds) - 1, figsize=(4.3 * (len(seeds) - 1), 4 * len(cs)), squeeze=False)
    for i, (x, y) in enumerate(cs):
        if not ((ct == x).any() and (ct == y).any()):
            continue
        ea = contrast(scores[a][common].values, ct, x, y)
        for k, b in enumerate(seeds[1:]):
            eb = contrast(scores[b][common].values, ct, x, y)
            ax = axes[i][k]
            ax.scatter(ea, eb, s=8, c="#1f77b4")
            lim = np.nanmax(np.abs(np.r_[ea, eb])) * 1.1
            ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5)
            ax.set_title(f"{x} vs {y}\nSpearman r = {spearmanr(ea, eb)[0]:.2f}", fontsize=9)
            ax.set_xlabel(f"{a} (SD)", fontsize=8), ax.set_ylabel(f"{b} (SD)", fontsize=8)
    fig.suptitle("Difference between two healthy cell types, per pathway (on the diagonal = same result)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_B0_contrasts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
