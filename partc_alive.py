"""
Part C: keep only the 'living' pathways and check the seed agreement again.

    python partc_alive.py --runs free=BASELINE_RUNS positive=FIX_RUNS/positive fixed=FIX_RUNS/fixed --out OUT

partc_diagnose.py showed that most pathways the model counts as active are 'dead': the encoder gives every
cell the same answer as the prior (KL near 0), so their scores are noise. A pathway is called alive in a
seed when its KL (averaged over the healthy reference cells) is at least --min_kl. The two groups are far
apart (dead about 1e-6, alive about 0.1 to 3), so the exact threshold hardly matters; the counts for
0.001, 0.01 and 0.1 are reported to show this.

Writes, per version, alive_<version>.csv (KL per seed, alive in all seeds yes/no) and the seed agreement on
the healthy cells for pathways alive in all seeds against the rest. The list for the baseline ('free')
version can be given to compare_seeds.py --keep to repeat the Part A seed check on living pathways only.
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

from partb_ref_seeds import contrast  # noqa: E402
from partc_diagnose import mean_var  # noqa: E402
from pathways_ctcl import pathway_scores  # noqa: E402
from run_experiment import prepare  # noqa: E402

CONTRASTS = [("CD4+ T cells", "CD14+ Monocytes"), ("CD4+ T cells", "CD8+ T cells")]


def agreement(scores, seeds, names, ct):
    """Seed agreement on the given pathways, averaged over the seed pairs."""
    if len(names) < 3:
        return {"pathways": len(names)}, np.array([])
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    r = np.array([[np.corrcoef(scores[a][n], scores[b][n])[0, 1] for n in names] for a, b in pairs])
    out = {"pathways": len(names), "median_cell_r": round(float(np.median(r)), 3),
           "share_cell_r_above_0.5": round(float((r > 0.5).mean()), 3)}
    for x, y in CONTRASTS:
        if (ct == x).any() and (ct == y).any():
            out[f"contrast_spearman[{x} vs {y}]"] = round(float(np.mean(
                [spearmanr(contrast(scores[a][names].values, ct, x, y),
                           contrast(scores[b][names].values, ct, x, y))[0] for a, b in pairs])), 3)
    return out, r.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder with seed0, seed1, seed2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--min_kl", type=float, default=0.01)
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    adata, _ = prepare(args.data, args.ref_max_cells, 0)
    ct = adata.obs["cell_type"].astype(str).values
    summary, hist = {"min_kl": args.min_kl}, {}
    for item in args.runs:
        name, folder = item.split("=", 1)
        seeds = sorted(d for d in os.listdir(folder)
                       if os.path.exists(os.path.join(folder, d, "ref_model", "model_params.pt")))
        scores, kl = {}, {}
        for s in seeds:
            path = os.path.join(folder, s, "ref_model")
            saved = np.genfromtxt(os.path.join(path, "var_names.csv"), delimiter=",", dtype=str)
            a = adata[:, list(saved)].copy()
            model = sca.models.EXPIMAP.load(path, a)
            mu, var = mean_var(model, a.X, a.obs["study"])
            sc_, names = pathway_scores(model, mu, a.uns["terms"])
            scores[s] = pd.DataFrame((sc_ - sc_.mean(0)) / (sc_.std(0) + 1e-8), columns=names)
            kl[s] = pd.Series((0.5 * (mu ** 2 + var - np.log(var) - 1)).mean(0), index=names)
            print(name, s, "active pathways:", len(names), flush=True)
        t = pd.DataFrame({f"kl_{s}": kl[s] for s in seeds})  # NaN = switched off by group lasso in that seed
        t["alive_all_seeds"] = (t[[f"kl_{s}" for s in seeds]] >= args.min_kl).all(1)
        t.to_csv(os.path.join(args.out, f"alive_{name}.csv"), index_label="pathway")

        common = sorted(set.intersection(*[set(scores[s].columns) for s in seeds]))
        alive = [n for n in common if t.loc[n, "alive_all_seeds"]]
        rest = [n for n in common if not t.loc[n, "alive_all_seeds"]]
        res = {"active_per_seed": {s: int(len(kl[s])) for s in seeds},
               "alive_per_seed": {s: int((kl[s] >= args.min_kl).sum()) for s in seeds},
               "dead_per_seed": {s: int((kl[s] < args.min_kl).sum()) for s in seeds},
               "active_in_all_seeds": len(common),
               "alive_in_all_seeds": len(alive),
               "alive_in_all_seeds_other_thresholds": {
                   str(th): int((t.loc[common, [f"kl_{s}" for s in seeds]] >= th).all(1).sum())
                   for th in (0.001, 0.01, 0.1)}}
        res["agreement_alive_in_all"], r_alive = agreement(scores, seeds, alive, ct)
        res["agreement_rest"], r_rest = agreement(scores, seeds, rest, ct)
        hist[name] = (r_alive, r_rest)
        summary[name] = res
        print(name, json.dumps(res), flush=True)

    json.dump(summary, open(os.path.join(args.out, "alive_summary.json"), "w"), indent=2)

    # figure: same-pathway agreement between seeds, living pathways against the rest
    fig, axes = plt.subplots(1, len(hist), figsize=(4 * len(hist), 3.2), squeeze=False)
    for ax, (n, (ra, rr)) in zip(axes[0], hist.items()):
        bins = np.linspace(-1, 1, 41)
        ax.hist(rr, bins=bins, color="#bbbbbb", label=f"dead in a seed ({len(rr)})")
        ax.hist(ra, bins=bins, color="#1f77b4", alpha=0.8, label=f"alive in all seeds ({len(ra)})")
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("same pathway, two seeds: correlation\n(mean over seed pairs)", fontsize=8)
        ax.set_ylabel("pathways", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_C_alive.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
