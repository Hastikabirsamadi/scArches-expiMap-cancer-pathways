"""
Part C, overlap test: does pathway activity move between pathways that share genes?

    python partc_overlap.py --runs free=BASELINE_RUNS positive=FIX_RUNS/positive fixed=FIX_RUNS/fixed --out OUT

Each runs folder holds seed0, seed1, seed2 with ref_model/ (as for compare_fix.py). No training: every
reference model scores the same healthy PBMC cells, then for every pair of seeds (a, b) and every pathway p
active in all seeds, three questions:
1. Is the information lost or only moved? R2 of seed a's score of p explained by ALL pathway scores of
   seed b together. High R2 with a low same-pathway correlation means seed b holds the same information but
   split differently over its pathways.
2. Does it move to overlapping pathways? R2 of p explained by seed b's p plus the k pathways that share
   most genes with p, against p plus k random pathways that share no gene with p.
3. Do pathways that overlap more with others disagree more? Spearman of the same-pathway correlation
   against the pathway's largest gene overlap (Jaccard) with another pathway, also after removing the
   effect of pathway size.
Caveat for 2: overlapping pathways may also move together for biological reasons, so a higher R2 there is
expected to some degree even in a perfectly stable model. The size of the gap is what matters.
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

from map_ctcl import latent  # noqa: E402
from pathways_ctcl import pathway_scores  # noqa: E402
from run_experiment import prepare  # noqa: E402


def r2(y, X):
    """Share of the variance of y (or of each column of y) explained by a linear fit on the columns of X."""
    X1 = np.c_[np.ones(len(X)), X]
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    return 1 - (res ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def load_version(folder, adata):
    """z-scored pathway scores of every seed's reference model, and the gene sets of its active pathways."""
    seeds = sorted(d for d in os.listdir(folder)
                   if os.path.exists(os.path.join(folder, d, "ref_model", "model_params.pt")))
    scores, sets = {}, {}
    for s in seeds:
        path = os.path.join(folder, s, "ref_model")
        saved = np.genfromtxt(os.path.join(path, "var_names.csv"), delimiter=",", dtype=str)
        a = adata[:, list(saved)].copy()
        model = sca.models.EXPIMAP.load(path, a)
        sc_, names = pathway_scores(model, latent(model, a.X, a.obs["study"]), a.uns["terms"])
        scores[s] = pd.DataFrame((sc_ - sc_.mean(0)) / (sc_.std(0) + 1e-8), columns=names)
        mask = model.model.decoder.L0.expr_L.mask.cpu().numpy()[:, model.nonzero_terms()]  # genes x pathways
        sets.update({n: frozenset(saved[mask[:, j] > 0]) for j, n in enumerate(names)})
        print(os.path.basename(folder), s, "active pathways:", len(names), flush=True)
    return seeds, scores, sets


def analyse(seeds, scores, sets, k, rng):
    common = sorted(set.intersection(*[set(scores[s].columns) for s in seeds]))
    G = [sets[n] for n in common]
    m = len(common)
    jac = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            inter = len(G[i] & G[j])
            jac[i, j] = jac[j, i] = inter / len(G[i] | G[j]) if inter else 0.0
    rows = []
    for ia, a in enumerate(seeds):
        for b in seeds[ia + 1:]:
            A, B = scores[a][common].values, scores[b][common].values
            r2_all = r2(A, B)  # every pathway of seed a against all pathways of seed b at once
            for p in range(m):
                r_self = np.corrcoef(A[:, p], B[:, p])[0, 1]
                nb = np.where(jac[p] > 0)[0]
                far = np.where((jac[p] == 0) & (np.arange(m) != p))[0]
                kk = min(k, len(nb), len(far))
                row = dict(pair=f"{a}_vs_{b}", pathway=common[p], genes=len(G[p]),
                           max_jaccard=jac[p].max(), n_overlapping=len(nb), r_self=r_self,
                           r2_self=r_self ** 2, r2_all=r2_all[p], k=kk)
                if kk:
                    top = nb[np.argsort(-jac[p, nb])[:kk]]
                    rnd = rng.choice(far, kk, replace=False)
                    row["r2_overlapping"] = r2(A[:, p], B[:, np.r_[p, top]])
                    row["r2_random"] = r2(A[:, p], B[:, np.r_[p, rnd]])
                rows.append(row)
    return pd.DataFrame(rows), m


def partial_spearman(x, y, z):
    """Spearman of x and y after removing the linear effect of z from the ranks of both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder with seed0, seed1, seed2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--k", type=int, default=5, help="overlapping (and random) pathways added in test 2")
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    adata, _ = prepare(args.data, args.ref_max_cells, 0)

    summary, per_all = {}, []
    for item in args.runs:
        name, folder = item.split("=", 1)
        seeds, scores, sets = load_version(folder, adata)
        per, m = analyse(seeds, scores, sets, args.k, np.random.RandomState(0))
        per.insert(0, "version", name)
        per_all.append(per)
        # one value per pathway: averaged over the seed pairs
        pw = per.groupby("pathway")[["r_self", "max_jaccard", "genes"]].mean()
        t = per.dropna(subset=["r2_overlapping"])
        summary[name] = {
            "pathways_active_in_all": m,
            "median_r2_same_pathway": round(float(per["r2_self"].median()), 3),
            "median_r2_all_pathways": round(float(per["r2_all"].median()), 3),
            "pathways_tested_overlap": int(t["pathway"].nunique()),
            f"median_r2_plus_{args.k}_overlapping": round(float(t["r2_overlapping"].median()), 3),
            f"median_r2_plus_{args.k}_random": round(float(t["r2_random"].median()), 3),
            "share_overlapping_beats_random": round(float((t["r2_overlapping"] > t["r2_random"]).mean()), 3),
            "spearman_r_self_vs_max_jaccard": round(float(spearmanr(pw["r_self"], pw["max_jaccard"])[0]), 3),
            "spearman_r_self_vs_max_jaccard_size_removed": round(
                partial_spearman(pw["r_self"], pw["max_jaccard"], pw["genes"]), 3),
            "spearman_r_self_vs_genes": round(float(spearmanr(pw["r_self"], pw["genes"])[0]), 3),
        }
        print(name, json.dumps(summary[name]), flush=True)

    per = pd.concat(per_all)
    per.to_csv(os.path.join(args.out, "overlap_per_pathway.csv"), index=False)
    json.dump(summary, open(os.path.join(args.out, "overlap_summary.json"), "w"), indent=2)

    # figure 1: how much of seed a's pathway score seed b can explain, from different sets of its pathways
    names = list(summary)
    cols = [("r2_self", "same pathway only"), ("r2_random", f"+ {args.k} random pathways (no shared gene)"),
            ("r2_overlapping", f"+ {args.k} pathways sharing most genes"), ("r2_all", "all pathways")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.2
    for i, (c, lab) in enumerate(cols):
        vals = [per.loc[per["version"] == n, c].median() for n in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("explained variance (median R²)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("A pathway score of one seed, explained by the pathway scores of another seed", fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_C_overlap_r2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # figure 2: same-pathway agreement against gene overlap, one dot per pathway
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), squeeze=False)
    for ax, n in zip(axes[0], names):
        pw = per[per["version"] == n].groupby("pathway")[["r_self", "max_jaccard"]].mean()
        ax.scatter(pw["max_jaccard"], pw["r_self"], s=8, c="#1f77b4")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("largest gene overlap with another pathway (Jaccard)", fontsize=8)
        ax.set_ylabel("same pathway, two seeds: correlation", fontsize=8)
        ax.set_title(f"{n}: Spearman {summary[n]['spearman_r_self_vs_max_jaccard']}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_C_overlap_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
