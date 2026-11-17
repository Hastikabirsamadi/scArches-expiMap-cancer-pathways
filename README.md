# expiMap on cancer data: which pathways change, and in which cells

This project uses expiMap, an interpretable deep learning model from scArches (Lotfollahi et al., *Nat. Biotechnol.*), to study cancer single-cell data. In expiMap every latent node stands for one Reactome pathway, so the model's output can be read directly as pathway activity. The plan is to train the model on healthy cells, map cancer cells onto it with architecture surgery, and ask which pathways change and in which cell types.

It continues my earlier course project, [Bioinformatics-course-project-scArches-](https://github.com/Hastikabirsamadi/Bioinformatics-course-project-scArches-). There I tested two changes to expiMap over 12 runs and found that pushing all pathway nodes apart damages the model's interpretability. Those 12 runs are used here as reference numbers.

The cancer studied is cutaneous T-cell lymphoma (CTCL), using the blood samples of the CTCL atlas on CELLxGENE (Sézary syndrome, mycosis fungoides and healthy donors). The healthy reference is the PBMC data from the scArches tutorial.

## Code
- `src/scarches/` is a copy of scArches v0.6.1 with the changes from the earlier project, and a new option `decoder_weights` that limits the gene weights inside each pathway (`free`, `positive` or `fixed`). It also has a new function `alive_terms`, which tells which pathways really carry information about the cells, since many pathways that expiMap counts as active are silent. All changes are switched off by default, so the model behaves like the original.
- `run_experiment.py` and `metrics.py` are the scripts from the earlier project. They prepare the reference data, train and map, and compute the mixing and label scores. `run_experiment.py` now also has the variants `positive` and `fixed` for Part C.
- `colab_cancer.ipynb` runs everything on Google Colab and saves all results to Google Drive.
- `prepare_ctcl.py` takes a subsample of the CTCL blood samples (at most 1000 cells per donor) without loading the whole 3.4 GB file into memory.
- `map_ctcl.py` maps the CTCL cells onto the healthy reference by surgery and checks how well the mapping worked.
- `plot_ctcl.py` draws the mapping figures from the saved results, one question per figure.
- `pathways_ctcl.py` is Part A. It finds the pathways that differ between malignant and normal CD4+ T cells of the same patients, and between MF and Sézary patients of the same studies. It also checks whether the top genes of each pathway change in the measured expression.
- `compare_seeds.py` repeats the comparison over three training seeds and keeps only the pathways that agree in all of them.
- `partb_ref_seeds.py` is Part B. It checks whether three reference models trained with different seeds give the same pathway scores to the same healthy cells, and whether they use the same genes for each pathway.
- `compare_fix.py` is Part C. It compares the original decoder with the two limited versions, on how well the seeds agree and on mapping quality.
- `partc_overlap.py` is also Part C. It checks whether a pathway's activity moves to other pathways that share genes with it when the model is trained again with another seed.
- `partc_diagnose.py` is also Part C. It asks what makes the few pathways that agree between seeds different from the rest: how much information they carry, how much the decoder uses them, and how well they follow cell types and the main directions of the data.
- `partc_alive.py` is also Part C. It separates the living pathways from the dead ones (the encoder gives every cell the same answer) and checks the seed agreement on the living ones. With `--keep`, `compare_seeds.py` then repeats the Part A seed check on the living pathways only.

## Part A result
CTCL blood cells (47,325 cells from 48 donors) were mapped onto the healthy PBMC reference by surgery. The mapping is good and stable over three seeds. About 90% of the authors' malignant cells land on reference CD4+ T cells, the median label confidence is 0.93, and these numbers barely change between seeds.

Healthy donors and patients come from different studies, so a difference between them could come from the lab as well as from the disease. Only comparisons inside the same patient were therefore tested. Malignant and normal CD4+ T cells of the same patient differ in 60 to 162 of about 240 pathways, depending on the seed. But the per-pathway effects do not agree between seeds (Spearman correlation close to 0), and only 7 pathways are significant with the same direction in all three seeds, close to the 5 expected by chance. Only lower NLR signalling (NUCLEOTIDE_BINDING_DOMAIN_LEUC) in malignant cells holds in every seed and is also visible in the measured genes.

MF and Sézary showed no difference. Early and late stage could not be compared, because every patient in the blood samples is late stage.

In short, expiMap places cells reliably, but its single pathway scores are not reproducible across training runs, so pathway lists from one run should not be reported as findings. Why this happens is the starting question of Part B.

## Part B result
Part B asks where the instability comes from. The three reference models of the baseline project (same healthy PBMC cells and settings, seeds 0, 1 and 2) scored the same 32,484 healthy cells, with no cancer data and no surgery. 214 pathways are active in all three models.

The models already disagree on healthy cells. For the same pathway in the same cells, the median correlation of the scores between two models is only 0.03 to 0.06, and only about 20% of pathways agree well (correlation above 0.5). Few pathways are simply flipped in sign (2 to 6% below -0.5). How much a pathway differs between two cell types agrees only partly between models, with a Spearman correlation of about 0.3 for CD4+ T cells vs CD14+ monocytes and 0.11 to 0.29 for CD4+ T vs CD8+ T cells. The smaller the difference, the less the models agree, and the cancer vs normal differences of Part A were small.

The decoder is one visible difference. Each model weights the genes of a pathway almost unrelatedly to the others, with a median cosine similarity of only 0.06 to 0.10. Pathways that group lasso almost switched off are not the reason, since there are only 2 to 4 per model and leaving them out changes nothing. At this point the gene weights looked like the cause, but Part C shows they are only part of it. Surgery keeps the reference decoder fixed, so whatever makes the reference models disagree carries over unchanged to the mapped cancer cells.

## Part C result
Part C first tried to fix the instability by limiting the gene weights inside each pathway, with three new reference models per version. Allowing only positive weights made the weights of a pathway more similar between seeds (median cosine 0.41 instead of 0.08), but the scores barely improved (median correlation on the same cells 0.08 instead of 0.04). Giving all genes of a pathway the same weight made the decoders identical, yet the scores agreed even less (0.02) and the mapping broke, with label transfer accuracy falling from 0.86 to 0.67. Since identical decoders still give disagreeing scores, the gene weights cannot be the whole cause.

A second test asked whether a run puts the same information under a different pathway that shares genes with the original one. All pathway scores of another seed together explain about 44% of a pathway's score, but that information is spread over many pathways, not moved to the overlapping ones. Gene overlap is only a small effect.

The real cause showed up when each pathway's stability was compared with how much information it carries. Most pathways that expiMap counts as active are dead. Their encoder gives every cell the same answer (the KL divergence from the prior is about 0.000002, against 0.1 to 3 for the others), and the decoder hardly uses them. This is the known VAE problem of posterior collapse. Their gene weights stay non-zero, so group lasso does not switch them off, and z-scoring the scores turns their tiny noise into numbers that look real. In the original model only 65 to 68 of about 240 to 250 active pathways per seed are alive, and 53 are alive in all three seeds.

The living pathways are stable. On the same healthy cells their median correlation between seeds is 0.89 instead of 0, and cell-type differences agree with a Spearman correlation of about 0.5 instead of 0. Repeating the Part A cancer comparison on the 53 living pathways gives effects that agree much better between seeds (Spearman 0.12 to 0.44 instead of about 0). Four pathways are lower in malignant than in normal CD4+ T cells of the same patients in every seed. These are NLR signalling (NUCLEOTIDE_BINDING_DOMAIN_LEUC), interleukin signalling, NOTCH signalling and biological oxidations, where about 0.4 would be expected by chance. The effects are small (0.2 to 0.65 SD), and NLR signalling is again the one with clear support in the measured genes.

As a solution, this repo's scArches has a new function `alive_terms`, which returns only the pathways that are active and carry information (mean KL of at least 0.01). With it, pathway results can be limited to the living pathways before they are reported. On the nine trained reference models it finds exactly the same living pathways as the Part C analysis. Adding it changes nothing else in scArches, and a small rerun of the full pipeline gave the same scores as before.

In short, expiMap's pathway scores are reproducible only for the pathways that stay alive during training, and any pathway result should be checked for this first. A next step would be to train in a way that keeps more pathways alive, for example with a weaker KL term or a minimum amount of information per pathway, and to test whether those pathways are then stable as well.
