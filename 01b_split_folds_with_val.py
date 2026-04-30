"""
Step 1b: Pre-split the tokenized Arrow dataset into per-fold train/val/test files.

Run this after 01_preprocess_and_tokenize.py and before 02_finetune.py.

Instead of filtering 2M cells at the start of each fine-tuning run, this
script does the filtering once upfront and saves per-fold datasets to disk.
Fine-tuning then just loads the right file directly.

Split proportions (donor-level, stratified by disease label):
    train : 70%  — gradient updates
    val   : 15%  — early stopping / checkpoint selection (never touches test)
    test  : 15%  — one-shot final evaluation only

The val split is carved out of the original train donors from cv_folds.pkl.
Test donors are unchanged. cv_folds.pkl is updated in place to record val
donor IDs so 02_finetune.py can log them.

Output structure:
    splits/
        cv_folds.pkl          (updated: now includes "val" key per fold)
        fold_0/
            train/            (Arrow dataset, ~70% of cells)
            val/              (Arrow dataset, ~15% of cells)
            test/             (Arrow dataset, ~15% of cells)
        fold_1/
            ...
        ...
"""

import pickle
import numpy as np
from pathlib import Path
from datasets import load_from_disk

# ── Config (must match 01_preprocess_and_tokenize.py) ────────────────────────
OUT_DIR     = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs")
TOKEN_DIR   = OUT_DIR / "tokenized"
SPLIT_DIR   = OUT_DIR / "splits"
N_PROC      = 32
RANDOM_SEED = 42

# Fraction of the non-test donors to hold out as validation.
# 70/15/15 target: if test ≈ 15% of all donors, the remaining 85% is the
# train pool. Holding out 17.65% of that pool gives 85% × 17.65% ≈ 15%
# of all donors for val, and 85% × 82.35% ≈ 70% for train.
VAL_FRAC = 15 / 85  # ≈ 0.1765

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load full tokenized dataset and CV fold definitions
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading tokenized dataset from {TOKEN_DIR / 'full'}...")
full_dataset = load_from_disk(str(TOKEN_DIR / "full"))
print(f"  Total cells: {len(full_dataset):,}")
print(f"  Features: {full_dataset.column_names}")

with open(SPLIT_DIR / "cv_folds.pkl", "rb") as f:
    folds = pickle.load(f)
print(f"  Loaded {len(folds)} CV folds\n")

rng = np.random.default_rng(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build donor-label lookup so we can stratify the val split
# ─────────────────────────────────────────────────────────────────────────────
# Build a {donor_id: label} map from the dataset itself so we don't need to
# reload the original h5ad.
print("Building donor → label lookup from tokenized dataset...")
donor_ids = full_dataset["donor_id"]
labels    = full_dataset["label"]
donor_label = {}
for d, l in zip(donor_ids, labels):
    donor_label[d] = l   # label is constant within a donor; last write wins
print(f"  Unique donors found: {len(donor_label):,}\n")


def stratified_val_split(train_donors: list, val_frac: float, rng) -> tuple[list, list]:
    """
    Split train_donors into (train_final, val) stratified by disease label.

    PD donors and healthy donors are shuffled independently, then val_frac
    of each group is held out for validation. This preserves the PD/healthy
    ratio in both sets.
    """
    pd_donors      = [d for d in train_donors if donor_label.get(d, 0) == 1]
    healthy_donors = [d for d in train_donors if donor_label.get(d, 0) == 0]

    pd_arr      = np.array(pd_donors)
    healthy_arr = np.array(healthy_donors)
    rng.shuffle(pd_arr)
    rng.shuffle(healthy_arr)

    n_val_pd      = max(1, round(len(pd_arr)      * val_frac))
    n_val_healthy = max(1, round(len(healthy_arr) * val_frac))

    val_donors   = np.concatenate([pd_arr[:n_val_pd],        healthy_arr[:n_val_healthy]])
    train_final  = np.concatenate([pd_arr[n_val_pd:],        healthy_arr[n_val_healthy:]])

    return train_final.tolist(), val_donors.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Filter and save each fold
# ─────────────────────────────────────────────────────────────────────────────
updated_folds = []

for fold_idx, fold in enumerate(folds):
    print(f"Fold {fold_idx + 1} / {len(folds)}")

    fold_dir = SPLIT_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(exist_ok=True)

    # Carve val out of the original train donors (test is untouched)
    test_donors = fold["test"]
    train_final, val_donors = stratified_val_split(fold["train"], VAL_FRAC, rng)

    train_set = set(train_final)
    val_set   = set(val_donors)
    test_set  = set(test_donors)

    # Sanity: no overlap between any pair of splits
    assert train_set.isdisjoint(val_set),  "Overlap between train and val donors!"
    assert train_set.isdisjoint(test_set), "Overlap between train and test donors!"
    assert val_set.isdisjoint(test_set),   "Overlap between val and test donors!"

    total = len(train_set) + len(val_set) + len(test_set)
    print(f"  Donors — train: {len(train_set)}, val: {len(val_set)}, "
          f"test: {len(test_set)}  (total: {total})")
    print(f"  Approx split — train: {len(train_set)/total*100:.1f}%  "
          f"val: {len(val_set)/total*100:.1f}%  "
          f"test: {len(test_set)/total*100:.1f}%")

    # Filter cells by donor membership
    train_ds = full_dataset.filter(
        lambda x: x["donor_id"] in train_set,
        num_proc=N_PROC,
        desc=f"  Fold {fold_idx} train",
    )
    val_ds = full_dataset.filter(
        lambda x: x["donor_id"] in val_set,
        num_proc=N_PROC,
        desc=f"  Fold {fold_idx} val",
    )
    test_ds = full_dataset.filter(
        lambda x: x["donor_id"] in test_set,
        num_proc=N_PROC,
        desc=f"  Fold {fold_idx} test",
    )

    train_ds.save_to_disk(str(fold_dir / "train"))
    val_ds.save_to_disk(str(fold_dir / "val"))
    test_ds.save_to_disk(str(fold_dir / "test"))

    # Sanity check label balance across all three splits
    def label_summary(ds, name):
        lbls = ds["label"]
        n_pd      = sum(lbls)
        n_healthy = len(lbls) - n_pd
        print(f"  {name:6s}: {len(ds):>8,} cells  "
              f"(PD={n_pd:,}, healthy={n_healthy:,})")

    label_summary(train_ds, "train")
    label_summary(val_ds,   "val")
    label_summary(test_ds,  "test")
    print()

    # Update fold record to include val donors
    updated_folds.append({
        "train": train_final,
        "val":   val_donors,
        "test":  test_donors,
    })

# Overwrite cv_folds.pkl with the updated records that now include "val"
fold_path = SPLIT_DIR / "cv_folds.pkl"
with open(fold_path, "wb") as f:
    pickle.dump(updated_folds, f)
print(f"Updated cv_folds.pkl saved → {fold_path}")
print("  (Each fold now contains 'train', 'val', and 'test' donor lists.)")

print("\nDone. Per-fold datasets saved to:", SPLIT_DIR)
print("Next step: python 02_finetune.py")