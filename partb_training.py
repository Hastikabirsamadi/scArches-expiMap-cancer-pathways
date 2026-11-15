"""
Part B1: three ways to put the CTCL cells on the healthy reference map, scored the same way.

    python partb_training.py --ctcl SUB --ref_model REF --map DRIVE/ctcl_map/seed0 --out DRIVE/partB1/seed0

1. none:    the trained reference model is used as it is. CTCL cells are given the label of the largest
            reference study, because the model has no batch label for them (zero training).
2. surgery: the query model from map_ctcl.py (only the new CTCL batch weights were trained).
3. retrain: a new expiMap is trained from scratch on reference + CTCL cells together (everything trained).

For each way: training time, label transfer (share of malignant cells called CD4+ T, confidence),
mixing of healthy CTCL donors with the reference (iLISI), how well the authors' cell labels separate in the
latent space (cell-type ASW, NMI, ARI), and Part A comparison 1 (number of significant pathways, and how
similar the pathway effects are to surgery's).
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
warnings.simplefilter("ignore")

import anndata as ad  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import scanpy as sc  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import scarches as sca  # noqa: E402
from map_ctcl import get_reference, latent, match_genes  # noqa: E402
from metrics import celltype_asw, cluster_scores, ilisi  # noqa: E402
from pathways_ctcl import bh, paired_p, unique_names, within_patient  # noqa: E402
from run_experiment import ALPHA, EARLY_STOP, prepare  # noqa: E402


def pathway_scores(model, z, terms):
    """Sign-corrected pathway scores (higher = more active) and the names of the active pathways."""
    tmp = ad.AnnData(np.zeros((1, 1)))
    model.latent_directions(adata=tmp)
    active = model.nonzero_terms()
    names = unique_names(np.array(model.update_terms(list(terms)))[active])
    return z * tmp.uns["directions"][active], names


def evaluate(name, z_ref, z_q, model, adata, qa, args, extra):
    obs = qa.obs
    dis = obs[args.disease_key].astype(str).values
    lab = obs[args.label_key].astype(str).values
    mal = pd.Series(lab).str.lower().str.contains("malignant").values
    patient = dis != args.normal_value
    norm = ~patient

    clf = KNeighborsClassifier(n_neighbors=15).fit(z_ref, adata.obs["cell_type"].astype(str).values)
    proba = clf.predict_proba(z_q)
    pred = clf.classes_[proba.argmax(1)]
    r = dict(method=name, **extra)
    r["malignant_called_cd4"] = float((pred[mal] == args.cd4_label).mean())
    r["median_confidence"] = float(np.median(proba.max(1)))

    rng = np.random.RandomState(args.seed)
    n = min(len(z_ref), int(norm.sum()))
    z = np.vstack([z_ref[rng.choice(len(z_ref), n, replace=False)], z_q[norm][rng.choice(int(norm.sum()), n,
                                                                                          replace=False)]])
    r["ilisi_ref_vs_ctcl_normal"] = ilisi(z, np.array(["ref"] * n + ["ctcl"] * n), args.seed)
    r["author_label_asw"] = celltype_asw(z_q, lab, args.seed)
    r["author_label_nmi"], r["author_label_ari"] = cluster_scores(z_q, lab, args.seed)

    scores, names = pathway_scores(model, z_q, adata.uns["terms"])
    Z = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    cd4 = (pred == args.cd4_label) & ~mal
    D, used = within_patient(Z, obs[args.donor_key].astype(str).values, patient, mal, cd4, args.min_cells)
    r["pathways_active"] = int(len(names))
    r["partA_patients"] = len(used)
    eff = pd.Series(dtype=float)
    if len(used) >= 5:
        fdr = bh(paired_p(D))
        r["partA_significant_fdr05"] = int((fdr < 0.05).sum())
        eff = pd.Series(D.mean(0), index=names)
    print(name, json.dumps(r), flush=True)
    return r, eff, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctcl", required=True)
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--map", required=True, help="map_ctcl.py output with query_model/ (the surgery result)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--batch_key", default="accession")
    ap.add_argument("--label_key", default="author_sub_cell_type")
    ap.add_argument("--disease_key", default="disease")
    ap.add_argument("--donor_key", default="donor_id")
    ap.add_argument("--normal_value", default="normal")
    ap.add_argument("--cd4_label", default="CD4+ T cells")
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    ap.add_argument("--ctcl_max_cells", type=int, default=0, help="only for quick tests")
    ap.add_argument("--methods", default="none,surgery,retrain")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sc.settings.n_jobs = numba.config.NUMBA_NUM_THREADS  # Colab allows only 2 threads
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    adata, _ = prepare(args.data, args.ref_max_cells, args.seed)
    ref, adata, _ = get_reference(args, adata)
    genes = list(adata.var_names)
    q = sc.read_h5ad(args.ctcl)
    if args.ctcl_max_cells:
        q = q[np.sort(np.random.RandomState(0).choice(q.n_obs, min(args.ctcl_max_cells, q.n_obs), replace=False))]
    X, _ = match_genes(q, genes)
    obs = q.obs.copy()
    obs["study"] = "CTCL_" + obs[args.batch_key].astype(str)
    qa = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    qa.uns["terms"] = adata.uns["terms"]
    del q

    rows, effects, latents = [], {}, {}
    methods = args.methods.split(",")

    if "none" in methods:
        big = adata.obs["study"].astype(str).value_counts().index[0]
        print("no training: CTCL cells get the reference study label", big, flush=True)
        z_ref = latent(ref, adata.X, adata.obs["study"])
        z_q = latent(ref, qa.X, [big] * qa.n_obs)
        r, effects["none"], _ = evaluate("none", z_ref, z_q, ref, adata, qa, args,
                                         dict(train_time_s=0.0, epochs=0, ctcl_label_used=big))
        rows.append(r)
        latents["none"] = (z_ref, z_q)

    if "surgery" in methods:
        qpath = os.path.join(args.map, "query_model")
        if not os.path.exists(os.path.join(qpath, "model_params.pt")):
            raise SystemExit(f"no surgery model at {qpath}; run map_ctcl.py first")
        if args.ctcl_max_cells:
            print("note: surgery model was trained on all CTCL cells; scored here on the subsample", flush=True)
        qry = sca.models.EXPIMAP.load(qpath, qa)
        mpath = os.path.join(args.map, "metrics.json")
        m = json.load(open(mpath)) if os.path.exists(mpath) else {}
        log = os.path.join(args.map, "query_train_log.csv")
        ep = int(pd.read_csv(log)["epoch_loss"].notna().sum()) if os.path.exists(log) else None
        z_ref = latent(qry, adata.X, adata.obs["study"])
        z_q = latent(qry, qa.X, qa.obs["study"])
        # map_ctcl.py's time_s also includes loading data and drawing UMAPs, so it is an upper bound
        r, effects["surgery"], _ = evaluate("surgery", z_ref, z_q, qry, adata, qa, args,
                                            dict(train_time_s=m.get("time_s"), epochs=ep))
        rows.append(r)
        latents["surgery"] = (z_ref, z_q)

    if "retrain" in methods:
        joint = ad.AnnData(X=sparse.vstack([sparse.csr_matrix(adata.X), qa.X]).tocsr().astype(np.float32),
                           obs=pd.DataFrame({"study": np.r_[adata.obs["study"].astype(str).values,
                                                            qa.obs["study"].astype(str).values]}),
                           var=pd.DataFrame(index=genes))
        joint.obs_names = [f"c{i}" for i in range(joint.n_obs)]
        joint.varm["I"] = adata.varm["I"]
        joint.uns["terms"] = adata.uns["terms"]
        t0 = time.time()
        full = sca.models.EXPIMAP(adata=joint, condition_key="study", hidden_layer_sizes=[256, 256, 256],
                                  recon_loss="nb", use_extra_layer=False, use_corr_penalty=False)
        full.train(n_epochs=args.epochs, alpha_epoch_anneal=100, alpha=ALPHA, alpha_kl=0.5, weight_decay=0.0,
                   early_stopping_kwargs=EARLY_STOP, use_early_stopping=True, monitor_only_val=False,
                   seed=args.seed)
        tt = round(time.time() - t0, 1)
        full.save(os.path.join(args.out, "retrain_model"), overwrite=True)
        nref = adata.n_obs
        z_all = latent(full, joint.X, joint.obs["study"])
        r, effects["retrain"], _ = evaluate("retrain", z_all[:nref], z_all[nref:], full, adata, qa, args,
                                            dict(train_time_s=tt, epochs=len(full.trainer.logs.get("epoch_loss", []))))
        rows.append(r)
        latents["retrain"] = (z_all[:nref], z_all[nref:])

    res = pd.DataFrame(rows).set_index("method")
    if "surgery" in effects and len(effects["surgery"]):
        for k, e in effects.items():
            common = e.index.intersection(effects["surgery"].index)
            if len(e) and len(common) > 2:
                res.loc[k, "partA_spearman_vs_surgery"] = float(spearmanr(e[common], effects["surgery"][common])[0])
                res.loc[k, "partA_pathways_shared_with_surgery"] = int(len(common))
    res.to_csv(os.path.join(args.out, "results.csv"))
    print(res.T.to_string())

    # figure 1: the scores side by side
    cols = [c for c in ["malignant_called_cd4", "median_confidence", "ilisi_ref_vs_ctcl_normal", "author_label_asw",
                        "author_label_nmi", "author_label_ari", "partA_spearman_vs_surgery"] if c in res]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3.2))
    colors = {"none": "#bbbbbb", "surgery": "#1f77b4", "retrain": "#ff7f0e"}
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.bar(res.index, res[c], color=[colors.get(i, "grey") for i in res.index])
        ax.set_title(c.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Three ways to map CTCL cells (higher = better in every panel)", y=1.04)
    fig.savefig(os.path.join(args.out, "fig_B1_scores.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # figure 2: UMAP of reference + healthy CTCL donors for each method
    norm = (qa.obs[args.disease_key].astype(str) == args.normal_value).values
    fig, axes = plt.subplots(1, len(latents), figsize=(5 * len(latents), 4.6), squeeze=False)
    for ax, (k, (zr, zq)) in zip(axes[0], latents.items()):
        u = ad.AnnData(np.zeros((len(zr) + norm.sum(), 1), dtype=np.float32))
        u.obsm["X"] = np.vstack([zr, zq[norm]])
        sc.pp.neighbors(u, use_rep="X", random_state=args.seed)
        sc.tl.umap(u, random_state=args.seed)
        xy = u.obsm["X_umap"]
        src = np.r_[np.zeros(len(zr)), np.ones(norm.sum())]
        o = np.random.RandomState(0).permutation(len(xy))
        ax.scatter(xy[o, 0], xy[o, 1], s=1, c=np.where(src[o] == 0, "#ff7f0e", "#1f77b4"), linewidths=0,
                   rasterized=True)
        ax.set_title(f"{k}: orange = reference, blue = healthy CTCL donors", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.savefig(os.path.join(args.out, "fig_B1_umaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
