"""
Subsample the CTCL atlas (CELLxGENE) to the PBMC samples, with at most N cells per donor.

    python prepare_ctcl.py --h5ad ctcl_blood.h5ad --out ctcl_pbmc_subsample.h5ad

The full file (532,152 cells, 3.4 GB) does not fit in Colab's RAM. So the cell labels (obs) are read
first, the cells to keep are chosen from them, and only those rows of the raw count matrix are read
from disk, in chunks. The output keeps all genes, raw counts in .X and gene symbols as var_names.
"""
import argparse

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset
from scipy import sparse


def need(df, col, what):
    if col not in df.columns:
        raise SystemExit(f"column '{col}' ({what}) not found. Available columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tissue_key", default="tissue1")
    ap.add_argument("--tissue_value", default="PBMC")
    ap.add_argument("--donor_key", default="donor_id")
    ap.add_argument("--per_donor", type=int, default=1000, help="max cells per donor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20000, help="rows read from disk at a time")
    args = ap.parse_args()

    with h5py.File(args.h5ad, "r") as f:
        obs = read_elem(f["obs"])
        need(obs, args.tissue_key, "tissue")
        need(obs, args.donor_key, "donor")
        print("cells in file:", len(obs))
        print(obs[args.tissue_key].value_counts().to_string())

        # 1. choose cells: PBMC only, then at most per_donor random cells from each donor
        rng = np.random.RandomState(args.seed)
        pbmc = np.where((obs[args.tissue_key].astype(str) == args.tissue_value).values)[0]
        donors = obs[args.donor_key].astype(str).values
        keep = []
        for d in np.unique(donors[pbmc]):
            idx = pbmc[donors[pbmc] == d]
            keep.append(rng.choice(idx, min(args.per_donor, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        print(f"kept {len(keep)} of {len(pbmc)} PBMC cells from {len(np.unique(donors[pbmc]))} donors")
        # which studies hold healthy and which hold cancer donors (if they never overlap, study and
        # disease are confounded and the per-study batch correction could remove disease signal)
        kept = obs.iloc[keep]
        for col in ("accession", "disease"):
            need(kept, col, "study / disease")
        print("donors per study and disease (kept cells):")
        print(kept.groupby(["accession", "disease"], observed=True)[args.donor_key].nunique().unstack(fill_value=0).to_string())

        # 2. raw counts: raw/X if it exists, otherwise X
        xkey = "raw/X" if "raw" in f and "X" in f["raw"] else "X"
        vkey = "raw/var" if xkey == "raw/X" else "var"
        var = read_elem(f[vkey])
        if "feature_name" not in var.columns and vkey == "raw/var":  # symbols may be stored in var only
            v2 = read_elem(f["var"])
            if "feature_name" in v2.columns:
                var["feature_name"] = v2["feature_name"].reindex(var.index).values
        X = sparse_dataset(f[xkey])
        parts = [X[keep[i:i + args.chunk]] for i in range(0, len(keep), args.chunk)]
        X = sparse.vstack(parts).tocsr()
        print("matrix from", xkey, "shape", X.shape)

    # 3. check these are raw counts (whole numbers, not normalised values)
    sample = X.data[: min(len(X.data), 1_000_000)]
    is_int = bool(np.allclose(sample, np.round(sample)))
    print("values are whole numbers (raw counts):", is_int)
    if not is_int:
        raise SystemExit("the matrix does not look like raw counts; stop and check the file")
    X.data = X.data.astype(np.float32)

    var = var.copy()
    var["ensembl_id"] = var.index.astype(str)
    symbols = var["feature_name"].astype(str) if "feature_name" in var.columns else var.index.astype(str)
    var.index = pd.Index(symbols.values)
    var.index.name = None
    out = ad.AnnData(X=X, obs=obs.iloc[keep].copy(), var=var)
    out.var_names_make_unique()
    out.uns["subsample"] = dict(tissue=args.tissue_value, per_donor=args.per_donor, seed=args.seed,
                                n_pbmc_in_file=int(len(pbmc)))
    out.write_h5ad(args.out, compression="gzip")
    print("saved", args.out, out.shape)


if __name__ == "__main__":
    main()
