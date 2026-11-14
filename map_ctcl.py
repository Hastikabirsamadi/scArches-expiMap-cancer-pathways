"""
Map the CTCL subsample onto the healthy PBMC expiMap reference by architecture surgery.

    python map_ctcl.py --ctcl ctcl_pbmc_subsample.h5ad --ref_model DRIVE/ref_model --out DRIVE/ctcl_map

1. Reference: loads the trained PBMC reference from --ref_model.
2. Genes: the CTCL genes are matched to the reference genes by symbol. Missing genes are set to zero
   and counted in the report.
3. Surgery: each CTCL study (--batch_key) becomes a new batch. Only the new batch weights are trained.
4. Checks: kNN label transfer of the reference cell types, a cross-table against the authors' labels,
   and batch mixing between the reference and the healthy CTCL donors.
Outputs in --out: metrics.json, crosstab_*.csv, missing_genes.txt, UMAP figures, query_model/, and
ctcl_mapped.h5ad (cell labels, latent, pathway scores; no expression matrix).
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
import scanpy as sc  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import scarches as sca  # noqa: E402
from metrics import ilisi  # noqa: E402
from run_experiment import ALPHA, EARLY_STOP, prepare  # noqa: E402


def get_reference(args, adata):
    path = args.ref_model
    if os.path.exists(os.path.join(path, "model_params.pt")):
        saved = np.genfromtxt(os.path.join(path, "var_names.csv"), delimiter=",", dtype=str)
        if set(saved) != set(adata.var_names):
            raise SystemExit("the saved reference model uses different genes than this preprocessing; stop")
        adata = adata[:, list(saved)].copy()  # same gene order as the saved model
        print("loading reference model from", path, flush=True)
        return sca.models.EXPIMAP.load(path, adata), adata, "loaded"
    print("no reference model found, training one (baseline settings)", flush=True)
    ref = sca.models.EXPIMAP(adata=adata, condition_key="study", hidden_layer_sizes=[256, 256, 256],
                             recon_loss="nb", use_extra_layer=False, use_corr_penalty=False)
    ref.train(n_epochs=args.epochs, alpha_epoch_anneal=100, alpha=ALPHA, alpha_kl=0.5, weight_decay=0.0,
              early_stopping_kwargs=EARLY_STOP, use_early_stopping=True, monitor_only_val=False, seed=args.seed)
    ref.save(path, overwrite=True)
    return ref, adata, "trained"


def match_genes(q, genes):
    """Return a CSR matrix with the query counts in the reference gene order (zeros for missing genes)."""
    pos = pd.Index(q.var_names).get_indexer(genes)
    found = pos >= 0
    sel = sparse.csr_matrix(q.X)[:, pos[found]]
    # a 0/1 matrix that moves each found gene to its reference column; the other columns stay zero
    n = int(found.sum())
    mapper = sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.where(found)[0])),
                               shape=(n, len(genes)))
    return (sel @ mapper).tocsr().astype(np.float32), [g for g, f in zip(genes, found) if not f]


def latent(model, X, c, chunk=10000):
    """Posterior-mean latent of the active terms. Rows are densified in chunks, because scArches cannot
    index a scipy sparse matrix with its torch batch indices."""
    c = np.asarray(c)
    parts = []
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        x = x.toarray() if sparse.issparse(x) else np.asarray(x)
        parts.append(model.get_latent(x, c[i:i + chunk], mean=True, only_active=True))
    return np.vstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctcl", required=True)
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--batch_key", default="accession")
    ap.add_argument("--label_key", default="author_sub_cell_type")
    ap.add_argument("--disease_key", default="disease")
    ap.add_argument("--normal_value", default="normal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ref_max_cells", type=int, default=0, help="only for quick tests")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    res = dict(seed=args.seed, device="cuda" if torch.cuda.is_available() else "cpu")

    # 1. reference
    adata, _ = prepare(args.data, args.ref_max_cells, args.seed)
    ref, adata, res["reference"] = get_reference(args, adata)
    genes = list(adata.var_names)

    # 2. query genes
    q = sc.read_h5ad(args.ctcl)
    for k in (args.batch_key, args.label_key, args.disease_key):
        if k not in q.obs.columns:
            raise SystemExit(f"column '{k}' not in the CTCL file. Columns: {list(q.obs.columns)}")
    X, missing = match_genes(q, genes)
    obs = q.obs.copy()
    if "study" in obs.columns:
        obs = obs.rename(columns={"study": "study_original"})
    obs["study"] = "CTCL_" + obs[args.batch_key].astype(str)
    clash = set(obs["study"]) & set(adata.obs["study"].astype(str))
    if clash:
        raise SystemExit(f"CTCL batch names clash with reference studies: {clash}")
    qa = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    qa.uns["terms"] = adata.uns["terms"]
    del q
    with open(os.path.join(args.out, "missing_genes.txt"), "w") as fh:
        fh.write("\n".join(missing))
    res.update(n_ref=int(adata.n_obs), n_ctcl=int(qa.n_obs), n_genes=len(genes), n_genes_missing=len(missing),
               ctcl_batches={k: int(v) for k, v in qa.obs["study"].value_counts().items()},
               ctcl_disease={k: int(v) for k, v in qa.obs[args.disease_key].astype(str).value_counts().items()})
    print(f"genes: {len(genes) - len(missing)} of {len(genes)} reference genes found in CTCL", flush=True)

    # 3. surgery: only the new CTCL batch weights are trained
    qry = sca.models.EXPIMAP.load_query_data(qa, ref)
    qry.train(n_epochs=args.epochs, alpha_epoch_anneal=100, weight_decay=0.0, alpha_kl=0.1,
              seed=args.seed, use_early_stopping=True)
    res["query_epochs"] = len(qry.trainer.logs.get("epoch_loss", []))
    pd.DataFrame({k: pd.Series(v) for k, v in qry.trainer.logs.items()}).to_csv(
        os.path.join(args.out, "query_train_log.csv"), index=False)
    qry.save(os.path.join(args.out, "query_model"), overwrite=True)

    # latent space for reference + CTCL cells (posterior mean)
    active = qry.nonzero_terms()
    z_ref = latent(qry, adata.X, adata.obs["study"])
    z_q = latent(qry, qa.X, qa.obs["study"])
    res["terms_active"] = int(np.sum(active))
    res["terms_kept_after_surgery"] = float((ref.nonzero_terms() & active).sum() / max(ref.nonzero_terms().sum(), 1))

    # 4a. label transfer: reference cell types -> CTCL cells
    clf = KNeighborsClassifier(n_neighbors=15).fit(z_ref, adata.obs["cell_type"].astype(str).values)
    proba = clf.predict_proba(z_q)
    qa.obs["pred_cell_type"] = clf.classes_[proba.argmax(1)]
    qa.obs["pred_confidence"] = proba.max(1)
    dis = qa.obs[args.disease_key].astype(str)
    for d in sorted(dis.unique()):
        m = (dis == d).values
        pd.crosstab(qa.obs[args.label_key].astype(str)[m], qa.obs["pred_cell_type"][m]).to_csv(
            os.path.join(args.out, f"crosstab_{d.replace(' ', '_')}.csv"))
    res["pred_confidence_median"] = {d: float(qa.obs["pred_confidence"][(dis == d).values].median())
                                     for d in sorted(dis.unique())}
    lab = qa.obs[args.label_key].astype(str)
    mal = lab.str.lower().str.contains("malignant").values
    if mal.any():
        res["malignant_pred_share"] = {k: round(float(v), 4) for k, v in
                                       qa.obs["pred_cell_type"][mal].value_counts(normalize=True).head(5).items()}

    # 4b. batch mixing: reference vs healthy CTCL donors (1 = fully mixed, 0 = separate)
    norm = (dis == args.normal_value).values
    if norm.any():
        # equal numbers from both sides, so a perfect mix can reach 1
        rng = np.random.RandomState(args.seed)
        n = min(len(z_ref), int(norm.sum()))
        z = np.vstack([z_ref[rng.choice(len(z_ref), n, replace=False)],
                       z_q[norm][rng.choice(int(norm.sum()), n, replace=False)]])
        src = np.array(["reference"] * n + ["ctcl_normal"] * n)
        res["ilisi_ref_vs_ctcl_normal"] = ilisi(z, src, args.seed)
    else:
        print("no healthy CTCL cells found with", args.disease_key, "==", args.normal_value)

    # pathway scores (sign-corrected so higher = more active), saved for Part A
    tmp = ad.AnnData(np.zeros((1, 1)))
    qry.latent_directions(adata=tmp)
    terms = np.array(qry.update_terms(list(adata.uns["terms"])))
    signs = tmp.uns["directions"][active]
    q_scores = z_q * signs

    # save joint cell table and UMAPs
    ref_obs = pd.DataFrame({"source": "reference", "study": adata.obs["study"].astype(str).values,
                            "cell_type": adata.obs["cell_type"].astype(str).values}, index=adata.obs_names)
    q_obs = qa.obs.copy()
    q_obs["source"] = "CTCL"
    q_obs["cell_type"] = q_obs["pred_cell_type"]
    joint = ad.AnnData(obs=pd.concat([ref_obs, q_obs], join="outer"))
    joint.obs_names_make_unique()
    joint.obsm["X_cvae"] = np.vstack([z_ref, z_q]).astype(np.float32)
    joint.uns["active_terms"] = terms[active].tolist()
    sc.pp.neighbors(joint, use_rep="X_cvae", random_state=args.seed)
    sc.tl.umap(joint, random_state=args.seed)
    for c in joint.obs.columns:
        if joint.obs[c].dtype == object:
            joint.obs[c] = joint.obs[c].astype(str)
    sc.settings.figdir = args.out
    sc.pl.umap(joint, color=["source", "study", "cell_type"], frameon=False, wspace=0.6, show=False,
               save="_joint.png")
    ctcl = joint[joint.obs["source"] == "CTCL"].copy()
    sc.pl.umap(ctcl, color=[args.disease_key, args.label_key, "pred_confidence"], frameon=False, wspace=0.6,
               show=False, save="_ctcl.png")
    qa_out = ad.AnnData(obs=qa.obs.copy())
    qa_out.obsm["X_pathways"] = q_scores.astype(np.float32)
    qa_out.obsm["X_umap"] = ctcl.obsm["X_umap"]
    qa_out.uns["pathways"] = terms[active].tolist()
    qa_out.write_h5ad(os.path.join(args.out, "ctcl_mapped.h5ad"), compression="gzip")
    joint.write_h5ad(os.path.join(args.out, "joint_latent.h5ad"), compression="gzip")

    res["time_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
