# Geneformer Fine-Tuning: Parkinson's Disease Classification

Fine-tunes [Geneformer](https://huggingface.co/ctheodoris/Geneformer) on your
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

### 3. GPU check
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
A GPU with ≥24 GB VRAM is recommended (A100/H100 ideal for 2M cells).
On a smaller GPU, reduce `BATCH_SIZE` in `02_finetune.py`.

---

## Pipeline

```
parkinsons.h5ad
      │
      ▼
01_preprocess_and_tokenize.py   ← QC, rank-order tokenization, CV fold splits
      │
      ▼  geneformer_finetune/tokenized/   (Arrow dataset)
         geneformer_finetune/splits/cv_folds.pkl
      │
      ▼
01b_split_folds_with_val.py   ← QC, rank-order tokenization, CV fold splits
02_smoke.py (to check that everything works with a small subset)
02_finetune_tracking.py                  ← Fine-tune with WeightedTrainer, 5-fold CV
      │
      ▼  geneformer_finetune/results/fold_{0..4}/best_model/
         geneformer_finetune/results/cv_metrics.csv
      │
      ▼
03_analyze_results.py           ← Per-fold metrics, patient-level aggregation,
                                   cell-type breakdown, attention gene scores
```

### Run each step
```bash
python 01_preprocess_and_tokenize.py

# Run all folds sequentially:
python 02_finetune.py

# Or submit one fold per GPU job (e.g. SLURM):
for i in 0 1 2 3 4; do
  sbatch --gres=gpu:1 --wrap="python 02_finetune.py --fold $i"
done

python 03_analyze_results.py
```

---

## Key Design Decisions

### Patient-level CV (not cell-level)
Cells from the same patient are highly correlated. Splitting by cell would
leak patient signal across train/test, producing inflated metrics that don't
generalize. The 5 folds hold out ~15 PD + ~5 healthy donors each.

### Class imbalance (75 PD : 25 healthy)
Handled two ways:
1. `WeightedTrainer` applies inverse-frequency loss weights (~3× for healthy cells)
2. AUROC/AUPR are the primary metrics (robust to imbalance vs. accuracy)

### Partial layer freezing (`FREEZE_LAYERS = 2`)
Freezes the embedding layer + bottom 2 transformer layers. This:
- Preserves low-level gene co-expression representations
- Reduces trainable parameters → less overfitting on small cohorts
- Speeds up training

Set `FREEZE_LAYERS = 0` for full fine-tuning if you have ≥50 patients/class.

### Raw counts required
Geneformer's tokenizer normalizes internally (to 10k UMI) then rank-orders
genes. Pass **raw** (un-normalized) counts. If `adata.X` is log-normalized,
the script uses `adata.raw` automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| OOM during tokenization | Lower `chunk_size` in `TranscriptomeTokenizer` |
| OOM during training | Reduce `BATCH_SIZE`, increase `GRAD_ACCUM` |
| Gene IDs not found | Ensure var_names are Ensembl IDs (ENSG…) |
| AUROC ≈ 0.5 after training | Check for batch effects; run Harmony first |
| `load_from_disk` fails | Check `TOKEN_DIR` path; re-run step 1 |

---

## Expected Results

On similar brain snRNA-seq Parkinson's datasets, Geneformer fine-tuning
typically achieves:
- **Cell-level AUROC**: 0.75 – 0.90 (varies by cell type)
- **Donor-aggregated AUROC**: 0.85 – 0.95
- Dopaminergic neurons and microglia tend to be the most discriminative cell types

If performance is low, check:
1. Batch effects between PD/healthy sample processing batches
2. Whether the `disease` label correlates with a technical covariate (e.g. `Brain_bank`)
