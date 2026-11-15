"""
Does limiting the gene weights inside each pathway make the pathway scores stable, and what does it cost?

    python compare_fix.py --runs free=BASELINE_RUNS positive=FIX_RUNS/positive fixed=FIX_RUNS/fixed --out OUT

Each runs folder holds seed0, seed1, seed2, each with metrics.json and ref_model/ from run_experiment.py
(--save_models). For every version:
- stability: partb_ref_seeds.py on its three reference models (same healthy cells, do the seeds agree?)
- mapping quality: the baseline project's scores, mean and spread over the three seeds
Note: in the 'fixed' version every gene of a pathway has the same weight by construction, so its gene-weight
similarity between seeds is 1 (or -1) automatically. For that version only the score agreement is meaningful.
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAPPING = {  # baseline project scores: what they mean, higher is better for all of them
    "terms_active_ref": "active pathways",
    "ref_ilisi": "reference: batch mixing (iLISI)",
    "ref_celltype_asw": "reference: cell types apart (ASW)",
    "ref_nmi": "reference: cell types apart (NMI)",
    "joint_ilisi": "with Kang query: batch mixing (iLISI)",
    "lt_accuracy": "Kang query: label transfer accuracy",
    "ifn_auc": "Kang query: interferon pathway finds stimulated cells (AUC)",
}
STABILITY = {
    "median_cell_r": "same cells, same pathway: score correlation (median)",
    "share_cell_r_above_0.5": "share of pathways that agree well (r > 0.5)",
    "contrast_spearman[CD4+ T cells vs CD14+ Monocytes]": "CD4+ T vs monocytes: same pathway differences (Spearman)",
    "contrast_spearman[CD4+ T cells vs CD8+ T cells]": "CD4+ T vs CD8+ T: same pathway differences (Spearman)",
    "median_weight_cosine": "same pathway: gene weight similarity (median cosine)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder with seed0, seed1, seed2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for item in args.runs:
        name, folder = item.split("=", 1)
        seeds = sorted(d for d in os.listdir(folder) if d.startswith("seed")
                       and os.path.exists(os.path.join(folder, d, "ref_model", "model_params.pt")))
        if len(seeds) < 2:
            raise SystemExit(f"{name}: need at least 2 seeds with ref_model in {folder}, found {seeds}")
        row = {"version": name, "seeds": len(seeds)}
        # mapping quality, mean and standard deviation over seeds
        m = pd.DataFrame([json.load(open(os.path.join(folder, s, "metrics.json"))) for s in seeds])
        for k in MAPPING:
            if k in m:
                row[k] = m[k].mean()
                row[k + "_sd"] = m[k].std()
        # stability over seeds (healthy reference cells), reusing the Part B check
        sdir = os.path.join(args.out, name)
        if not os.path.exists(os.path.join(sdir, "summary.json")):
            subprocess.run([sys.executable, os.path.join(HERE, "partb_ref_seeds.py"), "--data", args.data,
                            "--out", sdir, "--refs"] + [os.path.join(folder, s, "ref_model") for s in seeds],
                           check=True)
        st = json.load(open(os.path.join(sdir, "summary.json")))
        pairs = [v for k, v in st.items() if "_vs_" in k]
        for k in STABILITY:
            vals = [p[k] for p in pairs if k in p]
            if vals:
                row[k] = float(np.mean(vals))
        row["active_in_all_seeds"] = st["active_in_all"]
        rows.append(row)
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)

    res = pd.DataFrame(rows).set_index("version")
    res.to_csv(os.path.join(args.out, "fix_comparison.csv"))

    # figure: stability (top) and mapping quality (bottom), one bar per version
    colors = ["#bbbbbb", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for title, keys, fname in (("Do the seeds agree? (1 = same result)", STABILITY, "fig_fix_stability.png"),
                               ("Mapping quality (mean over seeds, bar = spread)", MAPPING, "fig_fix_mapping.png")):
        keys = [k for k in keys if k in res]
        fig, axes = plt.subplots(1, len(keys), figsize=(2.8 * len(keys), 3.4), squeeze=False)
        for ax, k in zip(axes[0], keys):
            err = res[k + "_sd"] if k + "_sd" in res else None
            ax.bar(res.index, res[k], yerr=err, color=colors[:len(res)], capsize=3)
            label = (STABILITY | MAPPING)[k]
            ax.set_title("\n".join([label[i:i + 28] for i in range(0, len(label), 28)]), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(title, y=1.08)
        fig.savefig(os.path.join(args.out, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
