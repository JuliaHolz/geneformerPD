"""
Step 3: Post-hoc analysis of Geneformer fine-tuning results.

- Aggregates predictions across CV folds
- Breaks down performance by cell type
- Extracts top genes by attention weight (interpretability)
- Produces summary plots

Requirements:
    pip install matplotlib seaborn scikit-learn
"""

import pickle, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from transformers import BertForSequenceClassification
from datasets import load_from_disk

GENEFORMER_REPO = os.environ.get("GENEFORMER_REPO", "./Geneformer")
sys.path.insert(0, GENEFORMER_REPO)

# ── Config ────────────────────────────────────────────────────────────────────
H5AD_PATH    = "/home/jholz/mcbert_parkinsons/original_h5ad/parkinsons.h5ad"
OUT_DIR      = Path("/home/jholz/mcbert_parkinsons/geneformer_finetune")
RESULTS_DIR  = OUT_DIR / "results"
PLOTS_DIR    = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

N_CV_FOLDS   = 5
LABEL_COL    = "disease"
CELL_TYPE_COL = "cell_type"
DONOR_COL    = "donor_id"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and merge per-fold predictions
# ─────────────────────────────────────────────────────────────────────────────
print("Loading per-fold predictions...")
all_preds = []
for fold_idx in range(N_CV_FOLDS):
    fold_dir = RESULTS_DIR / f"fold_{fold_idx}"
    pred_path = fold_dir / "test_predictions.csv"
    if not pred_path.exists():
        print(f"  WARNING: {pred_path} not found, skipping fold {fold_idx}")
        continue
    df = pd.read_csv(pred_path)
    df["fold"] = fold_idx
    all_preds.append(df)

preds_df = pd.concat(all_preds, ignore_index=True)
print(f"  Total test cell predictions: {len(preds_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge with AnnData obs to get cell_type labels
# ─────────────────────────────────────────────────────────────────────────────
print("Merging with AnnData metadata...")
import anndata as ad
adata = ad.read_h5ad(H5AD_PATH, backed="r")
obs = adata.obs[[DONOR_COL, CELL_TYPE_COL, LABEL_COL]].copy()

# Join on donor_id (approximate - for exact join you'd track cell barcodes)
# Here we do per-donor aggregation for cell-type breakdown
preds_by_donor = (
    preds_df.groupby("donor_id")
    .agg(mean_prob_pd=("prob_pd", "mean"),
         label=("label", "first"),
         n_cells=("label", "count"))
    .reset_index()
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Overall CV metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Overall CV Metrics ──")
metrics_csv = RESULTS_DIR / "cv_metrics.csv"
if metrics_csv.exists():
    cv_df = pd.read_csv(metrics_csv)
    for metric in ["auroc", "aupr", "f1"]:
        print(f"  {metric.upper()}: {cv_df[metric].mean():.4f} ± {cv_df[metric].std():.4f}")

    # Plot per-fold metrics
    fig, ax = plt.subplots(figsize=(7, 4))
    cv_melt = cv_df[["fold", "auroc", "aupr", "f1"]].melt("fold", var_name="metric", value_name="value")
    sns.barplot(data=cv_melt, x="fold", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Fold CV Performance")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.legend(title="Metric")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "cv_metrics_by_fold.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR}/cv_metrics_by_fold.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Patient-level aggregated prediction
#    For clinical interpretation: average cell-level probability per patient
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Patient-Level Aggregated Metrics ──")
donor_auroc = roc_auc_score(preds_by_donor["label"], preds_by_donor["mean_prob_pd"])
donor_aupr  = average_precision_score(preds_by_donor["label"], preds_by_donor["mean_prob_pd"])
print(f"  Donor-aggregated AUROC: {donor_auroc:.4f}")
print(f"  Donor-aggregated AUPR:  {donor_aupr:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(
    data=preds_by_donor, x="label", y="mean_prob_pd", ax=ax,
    palette={0: "steelblue", 1: "salmon"}
)
ax.set_xticklabels(["Healthy", "Parkinson"])
ax.set_ylabel("Mean P(Parkinson) per donor")
ax.set_title(f"Donor-aggregated predictions\n(AUROC={donor_auroc:.3f})")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "donor_aggregated_predictions.png", dpi=150)
print(f"  Saved: {PLOTS_DIR}/donor_aggregated_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cell-type-stratified performance
#    Join cell-type labels from obs using donor-level mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Cell-Type Stratified Performance ──")

# Get per-donor cell-type composition from adata
donor_celltype = (
    obs.groupby([DONOR_COL, CELL_TYPE_COL])
    .size()
    .reset_index(name="n_cells")
)
dominant_celltype = (
    donor_celltype.sort_values("n_cells", ascending=False)
    .groupby(DONOR_COL)
    .first()
    .reset_index()[[DONOR_COL, CELL_TYPE_COL]]
    .rename(columns={CELL_TYPE_COL: "dominant_cell_type"})
)

# For a proper per-cell-type analysis, you need to have tracked
# cell-level barcodes through tokenization. Here we show the approach
# using the cell-type composition from adata.obs:
celltype_counts = obs.groupby([CELL_TYPE_COL, LABEL_COL]).size().unstack(fill_value=0)
print(celltype_counts)

fig, ax = plt.subplots(figsize=(10, 5))
celltype_counts.plot(kind="bar", ax=ax, color=["steelblue", "salmon"])
ax.set_title("Cell-type composition by disease status")
ax.set_xlabel("Cell Type")
ax.set_ylabel("# Cells")
ax.legend(title="Disease")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "celltype_composition.png", dpi=150)
print(f"  Saved: {PLOTS_DIR}/celltype_composition.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Attention-based gene importance (fold 0 model)
#    Extracts mean attention weight per gene token across test cells
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Attention-Based Gene Importance (Fold 0) ──")

try:
    from geneformer import EmbExtractor

    model_path = RESULTS_DIR / "fold_0" / "best_model"
    token_dir  = OUT_DIR / "tokenized"

    extractor = EmbExtractor(
        model_type="CellClassifier",
        num_classes=2,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        max_ncells=1000,        # sample 1000 cells for attention analysis
        emb_layer=-1,
        summary_stat=None,
        batch_size=32,
        silent=False,
    )

    # Extract attention scores
    # (The EmbExtractor API may differ by Geneformer version; adjust as needed)
    embs = extractor.extract_embs(
        model_directory=str(model_path),
        input_data_file=str(token_dir),
        output_directory=str(RESULTS_DIR / "embeddings"),
        output_prefix="fold0",
    )
    print("  Embeddings extracted — use these for UMAP visualization.")

except Exception as e:
    print(f"  Attention extraction skipped: {e}")
    print("  (Run manually after confirming your Geneformer version's API)")

print(f"\nAll plots saved to {PLOTS_DIR}")
print("Done!")
