"""
Step 1: Preprocess Parkinson's h5ad and tokenize for Geneformer.

Geneformer expects cells represented as rank-ordered gene token sequences.
This script:
  1. Loads obs metadata only (backed mode) to build CV folds and cell filter
  2. Writes a filtered h5ad safely in chunks (never mutates the backed file)
  3. Adds the required ensembl_id column to var (safe, non-backed write)
  4. Verifies raw counts
  5. Tokenizes with TranscriptomeTokenizer

Requirements:
    pip install geneformer anndata scanpy datasets

Geneformer repo must be cloned and on PYTHONPATH:
    git clone https://huggingface.co/ctheodoris/Geneformer
    export GENEFORMER_REPO=/path/to/Geneformer
"""

import os
import gc
import sys
import math
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
import h5py
from datasets import concatenate_datasets, load_from_disk


# ── Geneformer imports ────────────────────────────────────────────────────────
GENEFORMER_REPO = os.environ.get("GENEFORMER_REPO", "./Geneformer")
sys.path.insert(0, GENEFORMER_REPO)
from geneformer import TranscriptomeTokenizer  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
H5AD_PATH    = "/home/jholz/mcbert_parkinsons/original_h5ad/parkinsons.h5ad"
OUT_DIR      = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs")
TOKEN_DIR    = OUT_DIR / "tokenized"
SPLIT_DIR    = OUT_DIR / "splits"
TMP_H5AD     = OUT_DIR / "filtered_for_tokenization.h5ad"

LABEL_COL    = "disease"
DONOR_COL    = "donor_id"
HEALTHY_VAL  = "normal"
DISEASE_VAL  = "Parkinson disease"
N_CV_FOLDS   = 5
RANDOM_SEED  = 42
CHUNK_SIZE   = 50_000   # cells loaded into RAM at once; reduce if OOM
N_PROC       = 8        # tokenizer parallelism; match to your CPU count

TOKEN_INP_DIR = "/orcd/compute/edsun/001/jholz/finetuning/inputs/tokenizer_input"
'''
for d in [TOKEN_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load obs metadata only (backed=r keeps X memory-mapped, obs is tiny)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading AnnData metadata (backed mode)...")
adata = ad.read_h5ad(H5AD_PATH, backed="r")
print(f"  Shape: {adata.shape}")
print(f"  Disease value counts:\n{adata.obs[LABEL_COL].value_counts()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.5 get raw counts
# ─────────────────────────────────────────────────────────────────────────────

print("Writing raw counts to disk via h5py (bypassing anndata/scipy)...")
raw_h5ad = OUT_DIR / "raw_counts.h5ad"

if not raw_h5ad.exists():
    # Find the raw group directly in the HDF5 file
    with h5py.File(H5AD_PATH, "r") as f:
        print(f"  HDF5 root keys: {list(f.keys())}")
        print(f"  raw keys: {list(f['raw'].keys())}")
        
        # Read the CSR components directly
        data    = f["raw/X/data"][:]
        indices = f["raw/X/indices"][:]
        indptr  = f["raw/X/indptr"][:]
        shape   = tuple(f["raw/X"].attrs.get("shape", adata.raw.X.shape))
        
        print(f"  data dtype: {data.dtype}, shape: {data.shape}")
        print(f"  data sample: {data[:10]}")

    X_raw = sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)
    del data, indices, indptr
    gc.collect()

    adata_raw = ad.AnnData(
        X=X_raw,
        obs=adata.obs.copy(),
        var=adata.raw.var.copy(),
    )
    del X_raw
    gc.collect()

    adata_raw.write_h5ad(raw_h5ad)
    del adata_raw
    gc.collect()
    print(f"  Saved → {raw_h5ad}")

adata_raw = ad.read_h5ad(raw_h5ad, backed="r")
print(f"  raw shape: {adata_raw.shape}, dtype: {adata_raw.X.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Filter obs to valid disease labels; build integer positional index
# ─────────────────────────────────────────────────────────────────────────────
print("Filtering cells...")
keep_mask  = adata.obs[LABEL_COL].isin([HEALTHY_VAL, DISEASE_VAL])
obs_keep   = adata.obs[keep_mask].copy()
obs_keep["label"]    = (obs_keep[LABEL_COL] == DISEASE_VAL).astype(int)
obs_keep["cell_idx"] = np.where(keep_mask)[0]   # integer positions in adata

print(f"  Kept {len(obs_keep):,} cells  "
      f"(healthy={(obs_keep['label']==0).sum():,}, "
      f"PD={(obs_keep['label']==1).sum():,})")
print(f"  Unique donors: {obs_keep[DONOR_COL].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build patient-level CV folds (stratified by disease status)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding patient-level CV folds...")

donor_meta     = (
    obs_keep.groupby(DONOR_COL)["label"]
    .first().reset_index()
    .rename(columns={"label": "donor_label"})
)
pd_donors      = donor_meta[donor_meta["donor_label"] == 1][DONOR_COL].values.copy()
healthy_donors = donor_meta[donor_meta["donor_label"] == 0][DONOR_COL].values.copy()

rng.shuffle(pd_donors)
rng.shuffle(healthy_donors)

pd_folds      = np.array_split(pd_donors,      N_CV_FOLDS)
healthy_folds = np.array_split(healthy_donors, N_CV_FOLDS)

folds = []
for i in range(N_CV_FOLDS):
    test_donors  = np.concatenate([pd_folds[i], healthy_folds[i]])
    train_donors = np.concatenate(
        [d for j, d in enumerate(pd_folds)      if j != i]
      + [d for j, d in enumerate(healthy_folds) if j != i]
    )
    folds.append({"train": train_donors.tolist(), "test": test_donors.tolist()})
    print(f"  Fold {i}: train={len(train_donors)} donors, "
          f"test={len(test_donors)} donors")

fold_path = SPLIT_DIR / "cv_folds.pkl"
with open(fold_path, "wb") as f:
    pickle.dump(folds, f)
print(f"  Saved folds → {fold_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write filtered h5ad in chunks
#
#    We never mutate the backed adata object — that causes segfaults.
#    Instead, load X chunk-by-chunk into memory, attach obs labels, and
#    concatenate into a fresh h5ad.
#
#    Peak RAM ≈ CHUNK_SIZE × n_genes × 4 bytes (sparse, so much less in
#    practice). At CHUNK_SIZE=50k and ~5% sparsity: ~1.5 GB per chunk.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Writing filtered h5ad in chunks of {CHUNK_SIZE:,} cells...")
print(f"  Output → {TMP_H5AD}")

orig_indices = obs_keep["cell_idx"].values
n_total      = len(orig_indices)
n_chunks     = math.ceil(n_total / CHUNK_SIZE)
chunk_paths  = []

gc.collect()
print("Converting adata.raw to AnnData for slicing...", flush=True)


for i in range(n_chunks):
    idx_slice  = orig_indices[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    obs_slice  = obs_keep.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    chunk = adata_raw[idx_slice].to_memory()
    chunk.obs["label"]   = obs_slice["label"].values
    chunk.obs[DONOR_COL] = obs_slice[DONOR_COL].values

    # Write each chunk to a temp file to avoid accumulating RAM
    chunk_path = OUT_DIR / f"_chunk_{i:04d}.h5ad"
    chunk.write_h5ad(chunk_path)
    chunk_paths.append(chunk_path)
    del chunk
    gc.collect()
    print(f"  Chunk {i+1}/{n_chunks}  ({len(idx_slice):,} cells)")

gc.collect()
# Concatenate all chunks into the final file
print("  Concatenating chunks...")
chunks = [ad.read_h5ad(p) for p in chunk_paths]
combined = ad.concat(chunks, join="outer")
del chunks
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Add ensembl_id column to var
#
#    Do this before writing the final file — safe because we're working on
#    an in-memory AnnData, not a backed one.
#    var_names are bare Ensembl IDs (confirmed: ENSG00000186827 etc.)
# ─────────────────────────────────────────────────────────────────────────────
print("\nAdding ensembl_id column to var...")
assert combined.var_names[0].startswith("ENSG"), (
    f"Expected Ensembl IDs in var_names, got: {combined.var_names[:3].tolist()}"
)
combined.var["ensembl_id"] = combined.var_names.astype(str)

# Strip version suffixes if present (e.g. ENSG00000139618.12 → ENSG00000139618)
if combined.var["ensembl_id"].str.contains(r"\.").any():
    combined.var["ensembl_id"] = combined.var["ensembl_id"].str.split(".").str[0]
    print("  Stripped Ensembl version suffixes")
print(f"  ensembl_id sample: {combined.var['ensembl_id'][:3].tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Verify raw counts (Geneformer normalizes internally; must not be log'd)
# ─────────────────────────────────────────────────────────────────────────────
print("\nVerifying raw counts...")
sample       = combined.X[:200]
sample_dense = sample.toarray() if sp.issparse(sample) else np.array(sample)

if sample_dense.min() < 0:
    raise ValueError(
        "Negative values in adata.X — Geneformer requires raw counts, "
        "not log-normalized data. Re-run using adata.raw.X."
    )
if sample_dense.max() < 20:
    print("  WARNING: max expression looks very low — verify adata.X is raw counts, "
          "not log-normalized.")
else:
    print(f"  OK (sample max={sample_dense.max():.0f}, min={sample_dense.min():.0f})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Write final h5ad and clean up chunk files
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nWriting final h5ad → {TMP_H5AD}")
combined.write_h5ad(TMP_H5AD)
del combined
gc.collect()

print("  Removing temp chunk files...")
for p in chunk_paths:
    p.unlink(missing_ok=True)

print(f"  Done. {n_total:,} cells written.\n")
'''
# ─────────────────────────────────────────────────────────────────────────────
# 8. Tokenize with Geneformer
#
#    TranscriptomeTokenizer reads the h5ad from disk, normalizes each cell
#    to 10k counts, rank-orders genes by normalized expression, and emits
#    the top 2048 (V1) or 4096 (V2) gene tokens per cell.
# ─────────────────────────────────────────────────────────────────────────────
print("Tokenizing..." + TOKEN_INP_DIR, flush=True)

# Auto-detect V2 (gc104M) vs V1 pickle files
gf_dir = Path(GENEFORMER_REPO) / "geneformer"

def find_pkl(candidates: list[str]) -> str:
    for name in candidates:
        p = gf_dir / name
        if p.exists():
            print(f"  Using {p.name}")
            return str(p)
    raise FileNotFoundError(
        f"None of {candidates} found in {gf_dir}.\n"
        "Run: cd $GENEFORMER_REPO && git lfs install && git lfs pull"
    )

gene_median_file = find_pkl([
    "gene_median_dictionary_gc104M.pkl",
    "gene_median_dictionary.pkl",
])
token_dict_file  = find_pkl([
    "token_dictionary_gc104M.pkl",
    "token_dictionary.pkl",
])

tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"label": "label", DONOR_COL: "donor_id"},
    gene_median_file=f"/orcd/compute/edsun/001/jholz/finetuning/Geneformer/geneformer/gene_median_dictionary_gc104M.pkl",
    token_dictionary_file=f"/orcd/compute/edsun/001/jholz/finetuning/Geneformer/geneformer/token_dictionary_gc104M.pkl",
    nproc=8,
    chunk_size=512,
)


SHARD_SIZE = 200_000   # cells per shard; tune down if still OOM

# Split the filtered h5ad into shards and tokenize each separately
tokenizer_h5ad = "/orcd/compute/edsun/001/jholz/finetuning/inputs/tokenizer_input/filtered_for_tokenization.h5ad"
shard_datasets = []
adata_for_sharding = ad.read_h5ad(tokenizer_h5ad)
n_cells  = len(adata_for_sharding)
n_shards = math.ceil(n_cells / SHARD_SIZE)
TOKENIZER_INPUT_DIR = OUT_DIR / "tokenizer_inp"
for shard_idx in range(n_shards):
    shard_path    = TOKENIZER_INPUT_DIR / f"shard_{shard_idx:04d}.h5ad"
    shard_out_dir = TOKENIZER_INPUT_DIR / f"shard_{shard_idx:04d}_tokenized"
    shard_out_dir.mkdir(exist_ok=True)
    
    # The tokenizer saves to: shard_out_dir / prefix / 
    prefix         = f"shard_{shard_idx:04d}"
    actual_out_dir = shard_out_dir / f"{prefix}.dataset"

    if not actual_out_dir.exists():
        start = shard_idx * SHARD_SIZE
        end   = min(start + SHARD_SIZE, n_cells)
        print(f"  Tokenizing shard {shard_idx+1}/{n_shards} (cells {start}–{end})...")
        shard = adata_for_sharding[start:end].copy()
        shard.write_h5ad(shard_path)

        tokenizer.tokenize_data(
            data_directory=str(TOKENIZER_INPUT_DIR),
            output_directory=str(shard_out_dir),
            output_prefix=prefix,
            file_format="h5ad",
        )
        shard_path.unlink(missing_ok=True)

    shard_datasets.append(load_from_disk(str(actual_out_dir)))
# Concatenate all shards into one dataset
print("Concatenating shards...")
full_dataset = concatenate_datasets(shard_datasets)
full_dataset.save_to_disk(str(TOKEN_DIR / "full"))
print(f"  Total cells tokenized: {len(full_dataset)}")
print(f"  Saved → {TOKEN_DIR}/full")

print(f"\nTokenization complete.")
print(f"  Arrow dataset  → {TOKEN_DIR}")
print(f"  CV fold splits → {SPLIT_DIR / 'cv_folds.pkl'}")
print("\nNext step: python 02_finetune.py")