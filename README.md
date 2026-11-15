# expiMap on cancer data: which pathways change, and in which cells

This project uses expiMap, an interpretable deep learning model from scArches (Lotfollahi et al., *Nat. Biotechnol.*), to study cancer single-cell data. In expiMap every latent node stands for one Reactome pathway, so the model's output can be read directly as pathway activity. The plan is to train the model on healthy cells, map cancer cells onto it with architecture surgery, and ask which pathways change and in which cell types.

It continues my earlier course project, [Bioinformatics-course-project-scArches-](https://github.com/Hastikabirsamadi/Bioinformatics-course-project-scArches-). There I tested two changes to expiMap over 12 runs and found that pushing all pathway nodes apart damages the model's interpretability. Those 12 runs are used here as reference numbers.

The cancer studied is cutaneous T-cell lymphoma (CTCL), using the blood samples of the CTCL atlas on CELLxGENE (Sézary syndrome, mycosis fungoides and healthy donors). The healthy reference is the PBMC data from the scArches tutorial.

## Code
- `src/scarches/` is a copy of scArches v0.6.1 with the changes from the earlier project. Both changes are switched off by default, so the model behaves like the original.
- `run_experiment.py` and `metrics.py` are the scripts from the earlier project. They prepare the reference data and compute the mixing and label scores.
- `colab_cancer.ipynb` runs everything on Google Colab and saves all results to Google Drive.
- `prepare_ctcl.py` takes a subsample of the CTCL blood samples (at most 1000 cells per donor) without loading the whole 3.4 GB file into memory.
- `map_ctcl.py` maps the CTCL cells onto the healthy reference by surgery and checks how well the mapping worked.
- `plot_ctcl.py` draws the mapping figures from the saved results, one question per figure.
- `pathways_ctcl.py` is Part A. It finds the pathways that differ between malignant and normal CD4+ T cells of the same patients, and between MF and Sézary patients of the same studies. It also checks whether the top genes of each pathway change in the measured expression.
- `compare_seeds.py` repeats the comparison over three training seeds and keeps only the pathways that agree in all of them.

## Part A result
CTCL blood cells (47,325 cells from 48 donors) were mapped onto the healthy PBMC reference by surgery. The mapping is good and stable over three seeds. About 90% of the authors' malignant cells land on reference CD4+ T cells, the median label confidence is 0.93, and these numbers barely change between seeds.

Healthy donors and patients come from different studies, so a difference between them could come from the lab as well as from the disease. Only comparisons inside the same patient were therefore tested. Malignant and normal CD4+ T cells of the same patient differ in 60 to 162 of about 240 pathways, depending on the seed. But the per-pathway effects do not agree between seeds (Spearman correlation close to 0), and only 7 pathways are significant with the same direction in all three seeds, close to the 5 expected by chance. Only lower NLR signalling (NUCLEOTIDE_BINDING_DOMAIN_LEUC) in malignant cells holds in every seed and is also visible in the measured genes.

MF and Sézary showed no difference. Early and late stage could not be compared, because every patient in the blood samples is late stage.

In short, expiMap places cells reliably, but its single pathway scores are not reproducible across training runs, so pathway lists from one run should not be reported as findings. Why this happens is the starting question of Part B.
