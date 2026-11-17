"""
Part C, diagnosis: what makes the few stable pathways different from the many unstable ones?

    python partc_diagnose.py --runs free=BASELINE_RUNS positive=FIX_RUNS/positive fixed=FIX_RUNS/fixed --out OUT

Each runs folder holds seed0, seed1, seed2 with ref_model/ (as for compare_fix.py). No training: every
reference model encodes the same healthy PBMC cells. Stability of a pathway = correlation of its score
between two seeds on the same cells, averaged over the seed pairs. It is compared with three properties,
each averaged over the seeds:
1. information (KL): how far the encoder moves a pathway away from the prior (mean 0, variance 1). A
   pathway with KL near 0 tells every cell "I don't know": its score is noise ("posterior collapse").
2. use in the decoder: share of the variation of the decoder's input to the genes (before the softmax) that
   comes from this pathway, var(score) x squared length of its gene weights. Near 0 = the model hardly
   uses it to rebuild the cells.
3. link to the main structure of the data: share of the pathway score explained by cell type (eta squared),
   and by the top 20 principal components of the measured expression. The script also reports how much of
   the expression variation the top 10, 20 and 50 components hold.
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
import scanpy as sc  # noqa: E402
import scarches as sca  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from partc_overlap import r2  # noqa: E402
from pathways_ctcl import pathway_scores  # noqa: E402
from run_experiment import prepare  # noqa: E402

N_PCS = 20


def mean_var(model, X, c, chunk=10000):
    """Encoder output for the active pathways: posterior mean and variance per cell."""
    c = np.asarray(c)
    mus, vs = [], []
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        x = x.toarray() if sparse.issparse(x) else np.asarray(x)
        mu, v = model.get_latent(x, c[i:i + chunk], mean=False, mean_var=True, only_active=True)
        mus.append(mu), vs.append(v)
    return np.vstack(mus), np.vstack(vs)


def eta2(y, groups):
    """Share of the variance of y (each column) explained by the group means."""
    total = ((y - y.mean(0)) ** 2).sum(0)
    between = sum((groups == g).sum() * (y[groups == g].mean(0) - y.mean(0)) ** 2 for g in np.unique(groups))
    return between / total


def per_seed(path, adata, ct, pcs):
    saved = np.genfromtxt(os.path.join(path, "var_names.csv"), delimiter=",", dtype=str)
    a = adata[:, list(saved)].copy()
    model = sca.models.EXPIMAP.load(path, a)
    mu, var = mean_var(model, a.X, a.obs["study"])
    s, names = pathway_scores(model, mu, a.uns["terms"])
    Z = (s - s.mean(0)) / (s.std(0) + 1e-8)
    kl = (0.5 * (mu ** 2 + var - np.log(var) - 1)).mean(0)  # per pathway, averaged over cells
    W = model.model.decoder.L0.expr_L.weight.data.cpu().numpy()[:, model.nonzero_terms()]
    use = mu.var(0) * (W ** 2).sum(0)
    props = pd.DataFrame({"kl": kl, "decoder_use_share": use / use.sum(), "eta2_cell_type": eta2(Z, ct),
                          "r2_top_pcs": r2(Z, pcs)}, index=names)
    return pd.DataFrame(Z, columns=names), props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder with seed0, seed1, seed2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    adata, _ = prepare(args.data, args.ref_max_cells, 0)
    ct = adata.obs["cell_type"].astype(str).values
    # principal components of the measured expression (log-normalised, the same 1972 genes)
    e = adata.copy()
    sc.pp.normalize_total(e)
    sc.pp.log1p(e)
    sc.pp.pca(e, n_comps=50, random_state=0)
    cum = np.cumsum(e.uns["pca"]["variance_ratio"])
    summary = {"expression_variance_in_top_pcs": {n: round(float(cum[n - 1]), 3) for n in (10, 20, 50)}}
    pcs = e.obsm["X_pca"][:, :N_PCS]

    measures = ["kl", "decoder_use_share", "eta2_cell_type", "r2_top_pcs"]
    tables = {}
    for item in args.runs:
        name, folder = item.split("=", 1)
        seeds = sorted(d for d in os.listdir(folder)
                       if os.path.exists(os.path.join(folder, d, "ref_model", "model_params.pt")))
        scores, props = {}, {}
        for s in seeds:
            scores[s], props[s] = per_seed(os.path.join(folder, s, "ref_model"), adata, ct, pcs)
            print(name, s, "active pathways:", scores[s].shape[1], flush=True)
        common = sorted(set.intersection(*[set(scores[s].columns) for s in seeds]))
        pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
        t = pd.DataFrame(index=common)
        t["stability"] = np.mean([[np.corrcoef(scores[a][p], scores[b][p])[0, 1] for p in common]
                                  for a, b in pairs], axis=0)
        for m in measures:
            t[m] = np.mean([props[s].loc[common, m].values for s in seeds], axis=0)
        t.insert(0, "version", name)
        tables[name] = t

        stable, unstable = t["stability"] > 0.5, t["stability"] < 0.2
        res = {"pathways": len(common), "stable_r_above_0.5": int(stable.sum()),
               "unstable_r_below_0.2": int(unstable.sum())}
        for m in measures:
            res[f"spearman_stability_vs_{m}"] = round(float(spearmanr(t["stability"], t[m])[0]), 3)
            res[f"median_{m}_stable"] = float(f"{t.loc[stable, m].median():.4g}")
            res[f"median_{m}_unstable"] = float(f"{t.loc[unstable, m].median():.4g}")
            if stable.sum() and unstable.sum():
                res[f"p_{m}_stable_vs_unstable"] = float(f"{mannwhitneyu(t.loc[stable, m], t.loc[unstable, m]).pvalue:.3g}")
        # how much of the decoder's work the stable group does together
        res["decoder_use_share_of_stable_group"] = round(float(t.loc[stable, "decoder_use_share"].sum()), 3)
        summary[name] = res
        print(name, json.dumps(res), flush=True)

    per = pd.concat(tables.values())
    per.to_csv(os.path.join(args.out, "diagnose_per_pathway.csv"), index_label="pathway")
    json.dump(summary, open(os.path.join(args.out, "diagnose_summary.json"), "w"), indent=2)

    # figure: stability (y) against each property (x), one row per version, one dot per pathway
    labels = {"kl": "information (KL, log scale)", "decoder_use_share": "use in the decoder (share, log scale)",
              "eta2_cell_type": "explained by cell type", "r2_top_pcs": f"explained by top {N_PCS} PCs"}
    names = list(tables)
    fig, axes = plt.subplots(len(names), len(measures), figsize=(3.6 * len(measures), 3 * len(names)),
                             squeeze=False)
    for i, n in enumerate(names):
        t = tables[n]
        for j, m in enumerate(measures):
            ax = axes[i][j]
            ax.scatter(t[m], t["stability"], s=7, c="#1f77b4")
            if m in ("kl", "decoder_use_share"):
                ax.set_xscale("log")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(f"{n}: Spearman {summary[n][f'spearman_stability_vs_{m}']}", fontsize=8)
            ax.set_xlabel(labels[m], fontsize=8)
            if j == 0:
                ax.set_ylabel("stability between seeds", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_C_diagnose.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
