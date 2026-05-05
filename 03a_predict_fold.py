"""
Step 3: Generate cell-level predictions from the best saved model for one fold.

Loads the best model from a specified directory, runs inference on the fold's
test set, and writes a .csv with one row per cell containing:
    - cell_id       : unique cell identifier
    - donor_id      : donor the cell came from
    - cell_type     : cell type annotation (for downstream per-celltype analysis)
    - fold          : fold index this prediction came from
    - label         : ground-truth binary label (0=healthy, 1=PD)
    - pred          : predicted binary label (argmax of logits)
    - prob_pd       : predicted probability of Parkinson's (softmax of logit[1])

Usage examples
--------------
# Minimal — infer test-set path from model path
python 03_predict_fold.py \
    --model_dir /path/to/run1/fold_0/best_model \
    --fold 0

# Override the test-set location explicitly
python 03_predict_fold.py \
    --model_dir /path/to/run1/fold_0/best_model \
    --test_dir  /path/to/splits/fold_0/test \
    --fold 0

# Custom output location
python 03_predict_fold.py \
    --model_dir /path/to/run1/fold_0/best_model \
    --fold 0 \
    --out_csv   /path/to/results/fold_0_predictions.csv

Expected dataset columns (from 01_preprocess_and_tokenize.py):
    input_ids, label, donor_id, cell_id, cell_type
    (cell_id / cell_type names can be overridden with --cell_id_col / --cell_type_col)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification

# ── Defaults (mirror 02_finetune_tracking.py) ────────────────────────────────
DEFAULT_GENEFORMER_REPO = "/orcd/compute/edsun/001/jholz/finetuning/Geneformer"
DEFAULT_SPLIT_DIR       = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs/splits")
DEFAULT_MAX_INPUT_SIZE  = 2048
DEFAULT_BATCH_SIZE      = 192   # inference only — can be larger than train batch

# Column names written by the tokenizer — override via CLI if yours differ
DEFAULT_CELL_ID_COL   = "cell_id"
DEFAULT_CELL_TYPE_COL = "cell_type"


# ─────────────────────────────────────────────────────────────────────────────
# Collator (identical logic to training, but no label requirement)
# ─────────────────────────────────────────────────────────────────────────────

class GeneformerCollator:
    """Truncate → cast int32→int64 → pad → attention mask."""

    def __init__(self, pad_token_id: int, max_len: int):
        self.pad_token_id = pad_token_id
        self.max_len      = max_len

    def __call__(self, batch):
        input_ids = [
            torch.tensor(
                list(item["input_ids"][: self.max_len]),
                dtype=torch.long,
            )
            for item in batch
        ]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_pad_token_id(geneformer_repo: str) -> int:
    token_dict_path = Path(geneformer_repo) / "geneformer" / "token_dictionary_gc104M.pkl"
    if not token_dict_path.exists():
        raise FileNotFoundError(
            f"Token dictionary not found at {token_dict_path}. "
            "Check --geneformer_repo points to your Geneformer clone."
        )
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)
    pad_id = token_dict.get("<pad>", 0)
    print(f"  Pad token id: {pad_id}")
    return pad_id


def resolve_test_dir(args) -> Path:
    """
    Return the test dataset directory.
    If --test_dir is supplied, use it directly.
    Otherwise derive from --split_dir and --fold:
        <split_dir>/fold_<fold>/test
    """
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        test_dir = Path(args.split_dir) / f"fold_{args.fold}" / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset directory not found: {test_dir}\n"
            "Use --test_dir to supply the path explicitly."
        )
    return test_dir


def resolve_out_csv(args, model_dir: Path) -> Path:
    if args.out_csv:
        return Path(args.out_csv)
    # Default: place CSV next to the model directory
    return model_dir.parent / f"fold_{args.fold}_test_predictions.csv"


def validate_columns(dataset, cell_id_col: str, cell_type_col: str):
    """Warn clearly if expected metadata columns are missing."""
    missing = []
    for col in [cell_id_col, cell_type_col]:
        if col not in dataset.column_names:
            missing.append(col)
    if missing:
        print(
            f"\n  WARNING: column(s) {missing} not found in dataset.\n"
            f"  Available columns: {dataset.column_names}\n"
            f"  Use --cell_id_col / --cell_type_col to specify the correct names,\n"
            f"  or those columns will be filled with None in the output CSV.\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(
    model: BertForSequenceClassification,
    dataset,
    collator: GeneformerCollator,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch inference over `dataset`.

    Returns
    -------
    logits : np.ndarray, shape (N, num_labels)
    labels : np.ndarray, shape (N,)
    """
    model.eval()
    all_logits = []
    all_labels = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = [dataset[i] for i in range(start, end)]

        collated = collator(batch)
        input_ids      = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)
        labels         = collated["labels"]          # keep on CPU

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu().float().numpy())
        all_labels.append(labels.numpy())

        if (start // batch_size) % 20 == 0:
            pct = end / n * 100
            print(f"    {end:>8,} / {n:,} cells  ({pct:.1f}%)", flush=True)

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate cell-level predictions from a saved Geneformer fold model."
    )

    # Required
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to the saved best_model directory (contains config.json + pytorch_model.bin).",
    )
    parser.add_argument(
        "--fold", type=int, required=True,
        help="Fold index (0-indexed) — recorded in the output CSV.",
    )

    # Dataset location (usually auto-resolved)
    parser.add_argument(
        "--test_dir", default=None,
        help=(
            "Path to the Arrow test dataset directory. "
            "Defaults to <split_dir>/fold_<fold>/test."
        ),
    )
    parser.add_argument(
        "--split_dir", default=str(DEFAULT_SPLIT_DIR),
        help=f"Root splits directory. Default: {DEFAULT_SPLIT_DIR}",
    )

    # Geneformer resources
    parser.add_argument(
        "--geneformer_repo", default=DEFAULT_GENEFORMER_REPO,
        help=f"Path to your Geneformer repository clone. Default: {DEFAULT_GENEFORMER_REPO}",
    )

    # Inference settings
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Per-device inference batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--max_input_size", type=int, default=DEFAULT_MAX_INPUT_SIZE,
        help=f"Sequence truncation length. Default: {DEFAULT_MAX_INPUT_SIZE}",
    )

    # Column name overrides
    parser.add_argument(
        "--cell_id_col", default=DEFAULT_CELL_ID_COL,
        help=f"Dataset column containing cell IDs. Default: '{DEFAULT_CELL_ID_COL}'",
    )
    parser.add_argument(
        "--cell_type_col", default=DEFAULT_CELL_TYPE_COL,
        help=f"Dataset column containing cell type labels. Default: '{DEFAULT_CELL_TYPE_COL}'",
    )

    # Output
    parser.add_argument(
        "--out_csv", default=None,
        help="Output CSV path. Defaults to <model_dir>/../fold_<fold>_test_predictions.csv",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"ERROR: model directory not found: {model_dir}")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    test_dir = resolve_test_dir(args)
    out_csv  = resolve_out_csv(args, model_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFold          : {args.fold}")
    print(f"Model dir     : {model_dir}")
    print(f"Test set dir  : {test_dir}")
    print(f"Output CSV    : {out_csv}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}\n")

    # ── Load resources ────────────────────────────────────────────────────────
    print("Loading pad token id...")
    pad_token_id = load_pad_token_id(args.geneformer_repo)

    print(f"Loading model from {model_dir} ...")
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"\nLoading test dataset from {test_dir} ...")
    test_ds = load_from_disk(str(test_dir))
    print(f"  Cells      : {len(test_ds):,}")
    print(f"  Columns    : {test_ds.column_names}")

    validate_columns(test_ds, args.cell_id_col, args.cell_type_col)

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = GeneformerCollator(pad_token_id=pad_token_id, max_len=args.max_input_size)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference (batch size={args.batch_size}) ...")
    logits, true_labels = run_inference(
        model, test_ds, collator, args.batch_size, device
    )

    # Trim to true dataset length (guards against any padding artefacts)
    n = len(test_ds)
    logits      = logits[:n]
    true_labels = true_labels[:n]

    probs      = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    print("\nAssembling output CSV ...")

    def safe_col(col_name: str):
        """Return column values or a list of None if the column is absent."""
        if col_name in test_ds.column_names:
            return test_ds[col_name]
        return [None] * n

    df = pd.DataFrame({
        "cell_id":   safe_col(args.cell_id_col),
        "donor_id":  test_ds["donor_id"],
        "cell_type": safe_col(args.cell_type_col),
        "fold":      args.fold,
        "label":     true_labels,
        "pred":      pred_labels,
        "prob_pd":   probs[:, 1],
    })

    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df):,} rows → {out_csv}")

    # ── Quick sanity summary ──────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    auroc = roc_auc_score(true_labels, probs[:, 1])
    aupr  = average_precision_score(true_labels, probs[:, 1])
    f1    = f1_score(true_labels, pred_labels, average="binary")
    print(f"\nTest set metrics (fold {args.fold}):")
    print(f"  AUROC : {auroc:.4f}")
    print(f"  AUPR  : {aupr:.4f}")
    print(f"  F1    : {f1:.4f}")

    if args.cell_type_col in test_ds.column_names:
        print("\nCell counts per cell type in test set:")
        print(df["cell_type"].value_counts().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
