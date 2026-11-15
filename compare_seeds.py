"""
Do the Part A results hold up when the model is trained again with another random seed?

    python compare_seeds.py --runs DRIVE/ctcl_map/seed0 DRIVE/ctcl_map/seed1 DRIVE/ctcl_map/seed2 \
                            --out DRIVE/ctcl_map/seed_comparison

Each run folder must contain partA/ from pathways_ctcl.py. A pathway counts as robust when it is
significant (FDR < 0.05) with the same direction in every seed. Pathways that a seed's model switched off
(group lasso) are missing in that seed and so cannot be robust.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fdr", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tabs, checks = {}, {}
    for r in args.runs:
        name = os.path.basename(os.path.normpath(r))
        tab = pd.read_csv(os.path.join(r, "partA", "comp1_malignant_vs_cd4.csv"))
        dup = tab["pathway"][tab["pathway"].duplicated()].unique()
        if len(dup):  # older runs: two pathways share a cut name and cannot be told apart, so drop both
            print(f"{name}: dropping pathway names that occur twice: {list(dup)}")
            tab = tab[~tab["pathway"].isin(dup)]
        tabs[name] = tab.set_index("pathway")
        f = os.path.join(r, "partA", "comp1_gene_check_summary.csv")
        if os.path.exists(f):
            g = pd.read_csv(f)
            checks[name] = g[~g["pathway"].duplicated(keep=False)].set_index("pathway")
    seeds = list(tabs)
    allp = sorted(set().union(*[t.index for t in tabs.values()]))
    common = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))

    t = pd.DataFrame(index=allp)
    for s in seeds:
        t[f"diff_{s}"] = tabs[s]["mean_diff_sd"]
        t[f"fdr_{s}"] = tabs[s]["fdr"]
    diffs = t[[f"diff_{s}" for s in seeds]]
    sig = (t[[f"fdr_{s}" for s in seeds]] < args.fdr).values & diffs.notna().values
    sign = np.sign(diffs.fillna(0).values)
    t["seeds_significant"] = sig.sum(1)
    t["same_direction_all"] = (np.abs(sign.sum(1)) == len(seeds))
    t["robust"] = (t["seeds_significant"] == len(seeds)) & t["same_direction_all"]
    t["direction"] = np.where(diffs.mean(1) > 0, "up in malignant", "down in malignant")
    t["mean_diff_over_seeds"] = diffs.mean(1)
    for s in checks:
        g = checks[s]
        t[f"genes_agree_{s}"] = (g["genes_agree"].astype(str) + "/" + g["genes"].astype(str)).reindex(t.index)
    t = t.sort_values(["robust", "seeds_significant", "mean_diff_over_seeds"],
                      key=lambda c: c.abs() if c.name == "mean_diff_over_seeds" else c, ascending=False)
    t.to_csv(os.path.join(args.out, "pathways_across_seeds.csv"))

    # agreement between seeds: rank correlation of the effect sizes on the pathways all seeds share
    corr = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            corr[f"{a}_vs_{b}"] = round(float(spearmanr(tabs[a].loc[common, "mean_diff_sd"],
                                                        tabs[b].loc[common, "mean_diff_sd"])[0]), 3)
    res = {"seeds": seeds, "pathways_per_seed": {s: int(len(tabs[s])) for s in seeds},
           "pathways_in_all_seeds": len(common),
           "significant_per_seed": {s: int((tabs[s]["fdr"] < args.fdr).sum()) for s in seeds},
           "robust": int(t["robust"].sum()),
           "robust_up": int((t["robust"] & (t["direction"] == "up in malignant")).sum()),
           "robust_down": int((t["robust"] & (t["direction"] == "down in malignant")).sum()),
           "spearman_effect_sizes": corr}
    json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
    print("\nrobust pathways (significant, same direction in every seed):")
    cols = ["direction", "mean_diff_over_seeds"] + [c for c in t.columns if c.startswith("genes_agree_")]
    print(t[t["robust"]][cols].to_string())

    # figure: effect size in each seed against seed 0, one dot per pathway
    if len(seeds) > 1:
        a = seeds[0]
        fig, axes = plt.subplots(1, len(seeds) - 1, figsize=(4.5 * (len(seeds) - 1), 4.2), squeeze=False)
        for ax, b in zip(axes[0], seeds[1:]):
            x, y = tabs[a].loc[common, "mean_diff_sd"], tabs[b].loc[common, "mean_diff_sd"]
            rob = t.loc[common, "robust"].values
            ax.scatter(x[~rob], y[~rob], s=8, c="#bbbbbb", label="other")
            ax.scatter(x[rob], y[rob], s=12, c="#d62728", label="robust in all seeds")
            lim = np.nanmax(np.abs(np.r_[x, y])) * 1.1
            ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5)
            ax.axhline(0, color="grey", lw=0.3), ax.axvline(0, color="grey", lw=0.3)
            ax.set_xlabel(f"{a}: malignant minus normal (SD)"), ax.set_ylabel(f"{b} (SD)")
            ax.set_title(f"Spearman r = {corr[f'{a}_vs_{b}']}", fontsize=9)
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.suptitle("Same pathway, different training seed (on the diagonal = same result)", y=1.03)
        fig.savefig(os.path.join(args.out, "fig_seed_agreement.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
