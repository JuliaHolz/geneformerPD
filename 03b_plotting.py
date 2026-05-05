import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import warnings
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

warnings.filterwarnings("ignore")

# ── 1. Load fold predictions ──────────────────────────────────────────────────
fold_files = [f"fold{i}_test_pred.csv" for i in range(5)]
df = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)

# ── 2. Load h5ad and extract Braak score ─────────────────────────────────────
adata = ad.read_h5ad("pdsinglecell.h5ad", backed = "r")

# cell_id is the obs index (rownames); Braak score is in "path_braak_lb"
braak_map = adata.obs["path_braak_lb"]
 
df["braak"] = df["cell_id"].map(adata.obs["path_braak_lb"])
df = df.dropna(subset=["braak", "prob_pd"])
 
rho, pval = stats.spearmanr(df["braak"], df["prob_pd"])
 
# ── Plot ──────────────────────────────────────────────────────────────────────
stages = sorted(df["braak"].unique())
data_by_stage = [df.loc[df["braak"] == s, "prob_pd"].values for s in stages]
n_by_stage = [len(d) for d in data_by_stage]
 
fig, ax = plt.subplots(figsize=(9, 5))
 
bp = ax.boxplot(
    data_by_stage,
    positions=range(len(stages)),
    showfliers=False,       # omit outlier dots — too many points
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    boxprops=dict(facecolor="steelblue", alpha=0.8),
    whiskerprops=dict(color="steelblue"),
    capprops=dict(color="steelblue"),
)
 
ax.set_xticks(range(len(stages)))
ax.set_xticklabels([f"{s}\n(n={n_by_stage[i]:,})" for i, s in enumerate(stages)], fontsize=9)
ax.set_xlabel("Braak Stage")
ax.set_ylabel("prob_pd")
ax.set_title("PD Probability by Braak Stage")
 
ax.text(
    0.02, 0.97,
    f"Spearman ρ = {rho:.3f}  p = {pval:.2e}",
    transform=ax.transAxes, va="top", ha="left", fontsize=9,
    bbox=dict(facecolor="white", edgecolor="lightgrey", alpha=0.8, pad=4),
)
 
plt.tight_layout()
plt.savefig("braak_prob_pd.png", dpi=180, bbox_inches="tight")
print("Saved → braak_prob_pd.png")
plt.show()
 







 # ── Aggregate to donor level ──────────────────────────────────────────────────
donor_df = (
    df.groupby("donor_id")
    .agg(mean_prob_pd=("prob_pd", "mean"), braak=("braak", "first"))
    .reset_index()
)
 
rho, pval = stats.spearmanr(donor_df["braak"], donor_df["mean_prob_pd"])
 
# ── Plot ──────────────────────────────────────────────────────────────────────
stages = sorted(donor_df["braak"].unique())
data_by_stage = [donor_df.loc[donor_df["braak"] == s, "mean_prob_pd"].values for s in stages]
n_by_stage = [len(d) for d in data_by_stage]
 
fig, ax = plt.subplots(figsize=(9, 5))
 
bp = ax.boxplot(
    data_by_stage,
    positions=range(len(stages)),
    showfliers=True,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    boxprops=dict(facecolor="steelblue", alpha=0.8),
    whiskerprops=dict(color="steelblue"),
    capprops=dict(color="steelblue"),
    flierprops=dict(marker="o", markerfacecolor="steelblue", markersize=5, alpha=0.5, linestyle="none"),
)
 
ax.set_xticks(range(len(stages)))
ax.set_xticklabels([f"{s}\n(n={n_by_stage[i]})" for i, s in enumerate(stages)], fontsize=9)
ax.set_xlabel("Braak Stage")
ax.set_ylabel("Mean prob_pd per donor")
ax.set_title("Donor-level Mean PD Probability by Braak Stage")
 
ax.text(
    0.02, 0.97,
    f"Spearman ρ = {rho:.3f}  p = {pval:.2e}",
    transform=ax.transAxes, va="top", ha="left", fontsize=9,
    bbox=dict(facecolor="white", edgecolor="lightgrey", alpha=0.8, pad=4),
)
 
plt.tight_layout()
plt.savefig("braak_prob_pd_donor.png", dpi=180, bbox_inches="tight")
print("Saved → braak_prob_pd_donor.png")
plt.show()


df["braak"] = df["cell_id"].map(adata.obs["path_braak_lb"])
df = df.dropna(subset=["braak", "prob_pd", "cell_type"])












 
# ── Pivot to cell type x Braak stage ─────────────────────────────────────────
df = df.dropna(subset=["braak", "prob_pd", "cell_type"])
 
# ── Pivot to cell type x Braak stage ─────────────────────────────────────────
heatmap_df = (
    df.groupby(["cell_type", "braak"])["prob_pd"]
    .mean()
    .unstack("braak")
)
heatmap_df = heatmap_df[sorted(heatmap_df.columns)]
heatmap_df = heatmap_df.loc[heatmap_df.mean(axis=1).sort_values(ascending=False).index]
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(5, len(heatmap_df) * 0.4)))
 
sns.heatmap(
    heatmap_df,
    ax=ax,
    cmap="RdYlBu_r",
    vmin=0, vmax=1,
    linewidths=0.4,
    linecolor="lightgrey",
    annot=True, fmt=".2f", annot_kws={"size": 8},
    cbar_kws={"label": "Mean prob_pd", "shrink": 0.6},
)
 
ax.set_xlabel("Braak Stage")
ax.set_ylabel("Cell Type")
ax.set_title("Mean PD Probability by Cell Type and Braak Stage")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
 
plt.tight_layout()
plt.savefig("celltype_braak_heatmap.png", dpi=180, bbox_inches="tight")
print("Saved → celltype_braak_heatmap.png")








df["braak"] = df["cell_id"].map(adata.obs["path_braak_lb"])
df = df.dropna(subset=["braak", "prob_pd", "cell_type"])
 
# ── Filter to high-confidence PD cells ───────────────────────────────────────
df = df[df["prob_pd"] > 0.5]
print(f"Cells remaining after prob_pd > 0.5 filter: {len(df):,}")
 
# ── Pivot to cell type x Braak stage ─────────────────────────────────────────
heatmap_df = (
    df.groupby(["cell_type", "braak"])["prob_pd"]
    .mean()
    .unstack("braak")
)
heatmap_df = heatmap_df[sorted(heatmap_df.columns)]
heatmap_df = heatmap_df.loc[heatmap_df.mean(axis=1).sort_values(ascending=False).index]
 
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(5, len(heatmap_df) * 0.4)))
 
sns.heatmap(
    heatmap_df,
    ax=ax,
    cmap="RdYlBu_r",
    vmin=0.5, vmax=1,
    linewidths=0.4,
    linecolor="lightgrey",
    annot=True, fmt=".2f", annot_kws={"size": 8},
    cbar_kws={"label": "Mean prob_pd", "shrink": 0.6},
)
 
ax.set_xlabel("Braak Stage")
ax.set_ylabel("Cell Type")
ax.set_title("Mean PD Probability by Cell Type and Braak Stage\n(cells with prob_pd > 0.5 only)")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
 
plt.tight_layout()
plt.savefig("celltype_braak_heatmap_highconf.png", dpi=180, bbox_inches="tight")
print("Saved → celltype_braak_heatmap_highconf.png")
