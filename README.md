# expiMap on cancer data: which pathways change, and in which cells

This project uses expiMap, an interpretable deep learning model from scArches (Lotfollahi et al., *Nat. Biotechnol.*), to study cancer single-cell data. In expiMap every latent node stands for one Reactome pathway, so the model's output can be read directly as pathway activity. The plan is to train the model on healthy cells, map cancer cells onto it with architecture surgery, and ask which pathways change and in which cell types.

It continues my earlier course project, [Bioinformatics-course-project-scArches-](https://github.com/Hastikabirsamadi/Bioinformatics-course-project-scArches-). There I tested two changes to expiMap over 12 runs and found that pushing all pathway nodes apart damages the model's interpretability. Those 12 runs are used here as reference numbers.

The first cancer is cutaneous T-cell lymphoma (CTCL), using the blood samples of the CTCL atlas on CELLxGENE (Sézary syndrome, mycosis fungoides and healthy donors). The healthy reference is the PBMC data from the scArches tutorial.

## Code
The folder `src/scarches/` is a copy of scArches v0.6.1 with the changes from the earlier project. Both changes are switched off by default, so the model behaves like the original. `run_experiment.py` and `metrics.py` are the scripts from the earlier project, and `colab_cancer.ipynb` runs everything on Google Colab. `prepare_ctcl.py` takes a subsample of the CTCL blood samples, and `map_ctcl.py` maps it onto the healthy reference and checks how well the mapping worked. `plot_ctcl.py` draws the figures from the saved results, one question per figure. `pathways_ctcl.py` is Part A. It finds the pathways that differ between malignant and normal CD4+ T cells of the same patients, and between MF and Sézary patients of the same studies, and lists the genes with the largest weights in those pathways. `compare_seeds.py` checks which of these results hold up with other random seeds.
