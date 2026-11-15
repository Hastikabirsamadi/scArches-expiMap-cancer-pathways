"""
Part A for CTCL: which pathways change in the cancer cells, and which genes drive them.

    python pathways_ctcl.py --map DRIVE/ctcl_map/seed0

Reads ctcl_mapped.h5ad (pathway scores per cell) and query_model/ (decoder weights) from --map.
Writes everything to --map/partA.

Healthy donors and patients come from different studies (study and disease are confounded), so only
comparisons inside the same patients or the same studies are made:
1. malignant vs non-malignant CD4+ T cells of the same patient (paired, one value per patient)
2. MF vs Sezary malignant cells, only in studies that hold both, after removing each study's average
Each patient counts once in the tests (not each cell), because cells of one patient are not independent.
Scores are z-scored per pathway over all CTCL cells, so differences are in standard deviations.
"""
import argparse
import json
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

DISEASE_COLORS = {"Sezary syndrome": "#e8744c", "mycosis fungoides": "#9b59b6", "normal": "#4c9be8"}
ABBR = {"Sezary syndrome": "SS", "mycosis fungoides": "MF"}


def bh(p):
    """Benjamini-Hochberg FDR: corrects p-values for testing many pathways at once."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(q)
    res[order] = np.minimum(ranked, 1)
    out[ok] = res
    return out


def short(name, n=45):
    name = str(name).replace("REACTOME_", "")
    return name if len(name) <= n else name[: n - 1] + "…"


def gene_weights(model_dir, n_pathways):
    """Decoder weights (genes x active pathways), signed so that a positive weight means the gene goes
    up when the pathway score goes up (same sign rule as expiMap's latent_directions)."""
    state = torch.load(os.path.join(model_dir, "model_params.pt"), map_location="cpu")
    W = state["decoder.L0.expr_L.weight"].numpy()  # genes x all terms (out-of-mask weights are zero)
    active = np.abs(W).sum(0) > 0
    W = W[:, active]
    if W.shape[1] != n_pathways:
        raise SystemExit(f"model has {W.shape[1]} active pathways but the mapped file has {n_pathways}; stop")
    genes = np.genfromtxt(os.path.join(model_dir, "var_names.csv"), delimiter=",", dtype=str)
    sign = np.sign(W.sum(0))
    sign[sign == 0] = 1
    return W * sign, genes


def unique_names(names):
    """Reactome names are cut to 30 characters, so two pathways can end up with the same name
    (REGULATION_OF_INSULIN_SECRETIO). Later copies get _2, _3, in the model's fixed term order."""
    seen, out = {}, []
    for n in map(str, names):
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return np.array(out)


def within_patient(Z, donor, patient, mal, cd4, min_cells):
    """One row per patient: mean score of malignant cells minus mean of that patient's normal CD4+ T cells.
    Patients with fewer than min_cells in either group are left out."""
    rows, used = [], []
    for p in sorted(set(donor[patient])):
        a = patient & (donor == p) & mal
        b = patient & (donor == p) & cd4
        if a.sum() >= min_cells and b.sum() >= min_cells:
            rows.append(Z[a].mean(0) - Z[b].mean(0))
            used.append(p)
    return (np.vstack(rows) if rows else np.zeros((0, Z.shape[1]))), used


def paired_p(D):
    """Wilcoxon signed-rank p-value per pathway (column): is the per-patient difference centred on zero?"""
    return np.array([wilcoxon(D[:, j]).pvalue if np.any(D[:, j] != 0) else 1.0 for j in range(D.shape[1])])


def gene_check(counts_path, cells, donor, used, mal, cd4, patient, show, up, W, genes, names, out, n_genes=8):
    """Do the measured genes agree with the pathway result? For the top genes of each shown pathway,
    compare their real expression (log-normalised counts) in malignant vs normal CD4+ T cells of the same
    patient. A gene 'agrees' when it moves in the direction its decoder weight predicts. If most top genes
    agree, the pathway result is visible in the data itself, not only in the model's latent space."""
    from scipy import sparse
    c = ad.read_h5ad(counts_path)
    if not pd.Index(cells).isin(c.obs_names).all():
        raise SystemExit("cells in the mapped file are missing from --counts; is it the same subsample?")
    c = c[pd.Index(cells)]
    X = sparse.csr_matrix(c.X)
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1
    X = X.tocsc()  # fast column (gene) access
    gidx = pd.Index(c.var_names)
    up_set = set(up["pathway"])
    rows = []
    for n in show["pathway"]:
        sign_path = 1 if n in up_set else -1
        w = W[:, list(names).index(n)]
        for r, i in enumerate(np.argsort(-np.abs(w))[:n_genes]):
            if w[i] == 0 or genes[i] not in gidx:
                continue
            x = np.log1p(np.asarray(X[:, gidx.get_loc(genes[i])].todense()).ravel() / lib * 1e4)
            diffs = np.array([x[patient & (donor == p) & mal].mean() - x[patient & (donor == p) & cd4].mean()
                              for p in used])
            expect = np.sign(w[i]) * sign_path
            rows.append(dict(pathway=n, direction="up in malignant" if sign_path > 0 else "down in malignant",
                             rank=r + 1, gene=genes[i], weight=float(w[i]), expected="up" if expect > 0 else "down",
                             mean_logexpr_diff=float(diffs.mean()),
                             patients_agree=int((np.sign(diffs) == expect).sum()), patients=len(used),
                             agrees=bool(np.sign(diffs.mean()) == expect)))
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(out, "comp1_gene_check.csv"), index=False)
    summ = g.groupby("pathway", sort=False).agg(direction=("direction", "first"), genes=("gene", "size"),
                                                 genes_agree=("agrees", "sum"))
    summ.to_csv(os.path.join(out, "comp1_gene_check_summary.csv"))
    print("gene check (top genes per pathway that move as predicted):")
    print(summ.to_string())

    # figure: 6 pathways per direction, one bar per gene
    pick = [p for p in summ.index if summ.loc[p, "direction"].startswith("up")][:6] + \
           [p for p in summ.index if summ.loc[p, "direction"].startswith("down")][:6]
    if pick:
        ncol = 6
        nrow = int(np.ceil(len(pick) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, n in zip(axes.ravel(), pick):
            s = g[g["pathway"] == n][::-1]
            ax.barh(s["gene"], s["mean_logexpr_diff"], color=np.where(s["agrees"], "#2ca02c", "#bbbbbb"))
            ax.axvline(0, color="black", lw=0.5)
            ax.set_title(("UP: " if summ.loc[n, "direction"].startswith("up") else "DOWN: ") + short(n, 28),
                         fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.suptitle("Real expression of each pathway's top genes: malignant minus normal CD4+ T cells "
                     "(mean over patients, log scale)\ngreen = moves as the pathway result predicts, "
                     "grey = does not", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_A5_gene_check.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {"gene_check_genes": int(len(g)), "gene_check_agree": int(g["agrees"].sum())}


def heatmap(ax, M, rows, cols, title, vmax=None, cmap="RdBu_r", label=""):
    vmax = vmax or np.nanpercentile(np.abs(M), 98)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xticks(range(len(cols)), cols, fontsize=7, rotation=90)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label(label, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="output folder of map_ctcl.py")
    ap.add_argument("--donor_key", default="donor_id")
    ap.add_argument("--study_key", default="accession")
    ap.add_argument("--label_key", default="author_sub_cell_type")
    ap.add_argument("--disease_key", default="disease")
    ap.add_argument("--normal_value", default="normal")
    ap.add_argument("--cd4_label", default="CD4+ T cells", help="voted reference type used as the control")
    ap.add_argument("--stage_keys", default="stage1,stage2", help="only printed and saved for now")
    ap.add_argument("--min_cells", type=int, default=20, help="min cells per group per patient")
    ap.add_argument("--exclude_types", default="Megakaryocyte progenitors",
                    help="voted cell types left out of the cell-type heatmap (comma separated)")
    ap.add_argument("--counts", default="", help="CTCL subsample with raw counts, for the gene check")
    ap.add_argument("--top", type=int, default=15, help="pathways shown per direction")
    args = ap.parse_args()
    out = os.path.join(args.map, "partA")
    os.makedirs(out, exist_ok=True)

    m = ad.read_h5ad(os.path.join(args.map, "ctcl_mapped.h5ad"))
    names = unique_names(m.uns["pathways"])
    P = np.asarray(m.obsm["X_pathways"], float)
    Z = (P - P.mean(0)) / (P.std(0) + 1e-8)  # standard deviations, so pathways are comparable
    obs = m.obs
    donor = obs[args.donor_key].astype(str).values
    study = obs[args.study_key].astype(str).values
    dis = obs[args.disease_key].astype(str).values
    mal = obs[args.label_key].astype(str).str.lower().str.contains("malignant").values
    patient = dis != args.normal_value
    cd4 = (obs["pred_cell_type"].astype(str).values == args.cd4_label) & ~mal
    W, genes = gene_weights(os.path.join(args.map, "query_model"), len(names))
    print(f"{m.n_obs} CTCL cells, {len(names)} pathways, {patient.sum()} cells from patients")

    # donor table (also shows the stage columns, to plan the early vs late comparison)
    stage_keys = [k for k in args.stage_keys.split(",") if k in obs.columns]
    d = pd.DataFrame({"donor": donor, "study": study, "disease": dis, "malignant": mal, "cd4_control": cd4})
    for k in stage_keys:
        d[k] = obs[k].astype(str).values
    donors = d.groupby("donor").agg(study=("study", "first"), disease=("disease", "first"),
                                    n_cells=("donor", "size"), n_malignant=("malignant", "sum"),
                                    n_cd4_control=("cd4_control", "sum"),
                                    **{k: (k, "first") for k in stage_keys})
    donors.to_csv(os.path.join(out, "donors.csv"))
    print(donors[donors["disease"] != args.normal_value].to_string())

    # 1. malignant vs non-malignant CD4+ T cells, inside each patient
    D, used = within_patient(Z, donor, patient, mal, cd4, args.min_cells)
    res = {"comp1_patients_used": len(used),
           "comp1_patients_skipped": sorted(set(donor[patient]) - set(used))}
    print(f"comparison 1: {len(used)} patients have >= {args.min_cells} malignant and control cells")
    if len(used) < 5:
        raise SystemExit("fewer than 5 patients usable for comparison 1; lower --min_cells or check the labels")
    pvals = paired_p(D)
    c1 = pd.DataFrame({"pathway": names, "mean_diff_sd": D.mean(0), "median_diff_sd": np.median(D, 0),
                       "patients_up": (D > 0).sum(0), "patients": len(used), "p": pvals, "fdr": bh(pvals)})
    c1 = c1.sort_values(["fdr", "mean_diff_sd"], key=lambda s: s if s.name == "fdr" else -s.abs())
    c1.to_csv(os.path.join(out, "comp1_malignant_vs_cd4.csv"), index=False)
    sig = c1[c1["fdr"] < 0.05]
    res["comp1_significant_fdr05"] = int(len(sig))
    res["comp1_up_fdr05"] = int((sig["mean_diff_sd"] > 0).sum())
    res["comp1_down_fdr05"] = int((sig["mean_diff_sd"] < 0).sum())
    up = sig[sig["mean_diff_sd"] > 0].nlargest(args.top, "mean_diff_sd")
    down = sig[sig["mean_diff_sd"] < 0].nsmallest(args.top, "mean_diff_sd")
    show = pd.concat([up, down])
    print(f"comparison 1: {len(sig)} pathways differ (FDR < 0.05): {len(up)} shown up, {len(down)} shown down")

    if len(show):
        # heatmap: pathways x patients, patients grouped by disease
        dz = pd.Series([donors.loc[p, "disease"] for p in used], index=used)
        order = dz.sort_values().index
        cols = [used.index(p) for p in order]
        idx = [list(names).index(n) for n in show["pathway"]]
        fig, ax = plt.subplots(figsize=(4 + 0.3 * len(used), 2 + 0.25 * len(show)))
        heatmap(ax, D[np.ix_(cols, idx)].T, [short(n) for n in show["pathway"]], list(order),
                "Malignant minus non-malignant CD4+ T cells, per patient\n(top: higher in malignant)",
                label="difference (SD)")
        for t, p in zip(ax.get_xticklabels(), order):  # patient names coloured by disease
            t.set_color(DISEASE_COLORS.get(dz[p], "grey"))
        ax.axhline(len(up) - 0.5, color="black", lw=1)
        h = [plt.Line2D([], [], marker="s", ls="", color=c, label=f"{k} (name colour)") for k, c in DISEASE_COLORS.items()
             if k in set(dz)]
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.15, 1.0), frameon=False, fontsize=8)
        fig.savefig(os.path.join(out, "fig_A1_malignant_vs_cd4_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # UMAPs of the two strongest pathways in each direction
        pick = list(up["pathway"][:2]) + list(down["pathway"][:2])
        xy = m.obsm["X_umap"]
        fig, axes = plt.subplots(1, len(pick) + 1, figsize=(4.5 * (len(pick) + 1), 4.2))
        axes[0].scatter(xy[:, 0], xy[:, 1], s=1, c="#dddddd", linewidths=0, rasterized=True)
        axes[0].scatter(xy[mal, 0], xy[mal, 1], s=1, c="#d62728", linewidths=0, rasterized=True)
        axes[0].set_title("malignant cells (red)", fontsize=9)
        for ax, n in zip(axes[1:], pick):
            v = Z[:, list(names).index(n)]
            lo, hi = np.percentile(v, [1, 99])
            o = np.argsort(v)
            sc_ = ax.scatter(xy[o, 0], xy[o, 1], s=1, c=np.clip(v[o], lo, hi), cmap="viridis", linewidths=0,
                             rasterized=True)
            ax.set_title(short(n, 40), fontsize=9)
            plt.colorbar(sc_, ax=ax, fraction=0.04).set_label("score (SD)", fontsize=7)
        for ax in axes:
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        fig.savefig(os.path.join(out, "fig_A2_pathway_umaps.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # top genes behind the shown pathways (candidates only: decoder weights, not a test)
        g = []
        for n in show["pathway"]:
            w = W[:, list(names).index(n)]
            for r, i in enumerate(np.argsort(-np.abs(w))[:10]):
                if w[i] != 0:
                    g.append(dict(pathway=n, direction="up in malignant" if n in set(up["pathway"]) else
                                  "down in malignant", rank=r + 1, gene=genes[i], weight=float(w[i])))
        pd.DataFrame(g).to_csv(os.path.join(out, "comp1_top_genes.csv"), index=False)
        if args.counts:
            res.update(gene_check(args.counts, m.obs_names, donor, used, mal, cd4, patient, show, up, W, genes,
                                  names, out))
        else:
            print("gene check skipped (no --counts file given)")

    # 3. pathway profile per cell type (patients only): which pathways mark which cell type
    ct = np.where(mal, "malignant", obs["pred_cell_type"].astype(str).values)
    # platelet-like cells (voted megakaryocyte progenitors) are left out: they have no nucleus and very little
    # RNA, so their scores are extreme and would take up the whole colour scale
    excl = [c.strip() for c in args.exclude_types.split(",") if c.strip()]
    keep_ct = [c for c, n in pd.Series(ct[patient]).value_counts().items() if n >= 200 and c not in excl]
    M = np.vstack([Z[patient & (ct == c)].mean(0) for c in keep_ct])  # cell types x pathways
    var_idx = np.argsort(-M.var(0))[:30]
    fig, ax = plt.subplots(figsize=(3 + 0.45 * len(keep_ct), 9))
    heatmap(ax, M[:, var_idx].T, [short(n) for n in names[var_idx]], keep_ct,
            "Mean pathway score per cell type (patients only)\n30 pathways that differ most between types",
            vmax=1.5, label="mean score (SD, capped at 1.5)")
    fig.savefig(os.path.join(out, "fig_A3_celltype_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. MF vs Sezary malignant cells, only in studies that hold both diseases
    pat = donors[(donors["disease"] != args.normal_value) & (donors["n_malignant"] >= args.min_cells)]
    both = [s for s, g in pat.groupby("study") if g["disease"].nunique() > 1]
    pat = pat[pat["study"].isin(both)]
    res["comp2_studies"] = both
    res["comp2_donors_per_disease"] = {k: int(v) for k, v in pat["disease"].value_counts().items()}
    print("comparison 2 uses studies", both, "donors:", res["comp2_donors_per_disease"])
    if pat["disease"].nunique() == 2 and pat["disease"].value_counts().min() >= 3:
        means = np.vstack([Z[patient & mal & (donor == p)].mean(0) for p in pat.index])
        means = means - pd.DataFrame(means).groupby(pat["study"].values).transform("mean").values
        a_name, b_name = sorted(pat["disease"].unique())
        a = means[(pat["disease"] == a_name).values]
        b = means[(pat["disease"] == b_name).values]
        p2 = np.array([mannwhitneyu(a[:, j], b[:, j]).pvalue for j in range(len(names))])
        c2 = pd.DataFrame({"pathway": names, f"mean_{a_name}_minus_{b_name}_sd": a.mean(0) - b.mean(0),
                           "p": p2, "fdr": bh(p2)})
        c2 = c2.sort_values(["p", c2.columns[1]], key=lambda s: s if s.name == "p" else -s.abs())
        c2.to_csv(os.path.join(out, "comp2_mf_vs_sezary.csv"), index=False)
        res["comp2_significant_fdr05"] = int((c2["fdr"] < 0.05).sum())
        res["comp2_lowest_p"] = float(c2["p"].min())
        # strip plot of the 6 pathways with the lowest p, one dot per patient
        top6 = c2["pathway"][:6].tolist()
        fig, axes = plt.subplots(1, 6, figsize=(16, 3.6), sharey=True)
        scol = dict(zip(both, plt.get_cmap("tab10").colors))
        for ax, n in zip(axes, top6):
            j = list(names).index(n)
            for k, (nm, grp) in enumerate([(a_name, a), (b_name, b)]):
                st = pat["study"].values[(pat["disease"] == nm).values]
                ax.scatter(np.full(len(grp), k) + np.random.RandomState(0).uniform(-0.12, 0.12, len(grp)),
                           grp[:, j], c=[scol[s] for s in st], s=25)
            ax.set_xticks([0, 1], [ABBR.get(a_name, a_name), ABBR.get(b_name, b_name)])
            ax.set_title(short(n, 30), fontsize=8)
            ax.axhline(0, color="grey", lw=0.5)
        axes[0].set_ylabel("malignant cells, donor mean\n(study average removed, SD)")
        h = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in scol.items()]
        fig.legend(handles=h, loc="upper right", frameon=False, fontsize=8)
        fig.suptitle("MF vs Sezary (SS) malignant cells: 6 lowest p-values (one dot = one patient)", y=1.05)
        fig.savefig(os.path.join(out, "fig_A4_mf_vs_sezary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("comparison 2 skipped: needs at least 3 patients of each disease in shared studies")

    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
