# Geneformer Fine-Tuning: Parkinson's Disease Classification

Fine-tunes [Geneformer](https://huggingface.co/ctheodoris/Geneformer) on a
Parkinson's single-cell dataset using patient-level cross-validation.

---

## Setup

### 1. Clone Geneformer
```bash
git clone https://huggingface.co/ctheodoris/Geneformer
export GENEFORMER_REPO=$(pwd)/Geneformer
```

### 2. Install dependencies
```bash
pip install torch transformers datasets accelerate scikit-learn \
            anndata scanpy matplotlib seaborn
pip install -e $GENEFORMER_REPO   # installs geneformer package
```

---

## Pipeline

1. Run 01a_preprocess_and_tokenize.py to tokenize the dataset using the scheme necessary for geneformer (ordered sequence of gene tokens)
2. Run 01b_split_folds_with_val.py to split the folds into separate files to improve parallelization and ease of finetuning.
3. Run 02_finetune.py to do finetuning. You can specify a fold with --fold option (0-4 in our case) to train a model for one fold to allow for parallel runs, otherwise this will run folds in serial. This should take about 30hrs/fold on an NVIDIA A100 GPU for the full dataset.
4. Run 03a_predict_fold.py to predict on the test set (again specifying fold, and best model location because we slightly changed the saving strategy after folds 0 and 1 to deal with jobs getting preempted.
5. Run 03b_plotting.py to generate figures.




