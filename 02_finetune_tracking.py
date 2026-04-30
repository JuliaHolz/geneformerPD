"""
Step 2: Fine-tune Geneformer for Parkinson's vs. Healthy classification.

Runs N_CV_FOLDS fine-tuning runs, each holding out a disjoint set of donors
as the test set. Saves per-fold checkpoints and metrics.

Split layout per fold:
  - train (70%): gradient updates
  - val   (20%): early stopping and checkpoint selection
  - test  (10%): final one-shot evaluation, never seen during training

Supports:
  - Automatic resumption from the latest checkpoint on preemption/restart
  - W&B experiment tracking (set WANDB_PROJECT env var or edit WANDB_PROJECT below)
  - Config logging at the start of each fold

Requirements:
    pip install transformers datasets accelerate scikit-learn torch geneformer wandb

Usage:
    python 02_finetune_with_val.py [--fold 0]   # run a single fold (for parallelism)
    python 02_finetune_with_val.py              # run all folds sequentially

NOTE: 01_split.py must produce train/, val/, and test/ sub-directories
under each fold directory (not just train/ and test/).  The simplest way
to add the val split is to further split the per-fold training donors
~78/22, which yields 70/20 of the full cohort once the 10% test hold-out
is removed.
"""

import os, sys, pickle, argparse, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback

import wandb


# ── Config ────────────────────────────────────────────────────────────────────
DONOR_COL        = "donor_id"
N_PROC           = 8
GENEFORMER_REPO  = "/orcd/compute/edsun/001/jholz/finetuning/Geneformer"

OUT_DIR          = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs")
TOKEN_DIR        = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs/tokenized/full")
SPLIT_DIR        = OUT_DIR / "splits"
RESULTS_DIR      = OUT_DIR / "run1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Geneformer pretrained weights (HuggingFace hub ID or local path)
PRETRAINED_MODEL = "/orcd/compute/edsun/001/jholz/finetuning/Geneformer/Geneformer-V2-104M"

# W&B project name — override with WANDB_PROJECT env var if preferred
WANDB_PROJECT    = "geneformerPD"

# Fine-tuning hyperparameters
MAX_INPUT_SIZE   = 2048   # Geneformer's context length
NUM_LABELS       = 2
BATCH_SIZE       = 96     # per GPU; reduce if OOM -- note initial was batch size 32, grad_accum 2
GRAD_ACCUM       = 1      # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_EPOCHS       = 5
LR               = 5e-5
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 0.01
FREEZE_LAYERS    = 6      # freeze bottom N transformer layers (0 = full fine-tune)
FP16             = torch.cuda.is_available()
N_CV_FOLDS       = 5
RANDOM_SEED      = 42

# Consolidated config dict — logged to W&B and saved as JSON at fold start
TRAIN_CONFIG = {
    "pretrained_model":  PRETRAINED_MODEL,
    "max_input_size":    MAX_INPUT_SIZE,
    "num_labels":        NUM_LABELS,
    "batch_size":        BATCH_SIZE,
    "grad_accum_steps":  GRAD_ACCUM,
    "effective_batch":   BATCH_SIZE * GRAD_ACCUM,
    "max_epochs":        MAX_EPOCHS,
    "learning_rate":     LR,
    "warmup_ratio":      WARMUP_RATIO,
    "weight_decay":      WEIGHT_DECAY,
    "freeze_layers":     FREEZE_LAYERS,
    "fp16":              FP16,
    "n_cv_folds":        N_CV_FOLDS,
    "random_seed":       RANDOM_SEED,
    "geneformer_repo":   GENEFORMER_REPO,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ADD at module level, near the other helper classes:
class GeneformerCollator:
    """
    Collator for Geneformer tokenized datasets.
    - Truncates to max_len tokens
    - Casts int32 gene token ids to int64 (required by nn.Embedding)
    - Pads to the longest sequence in the batch
    - Builds an attention mask
    """
    def __init__(self, pad_token_id: int, max_len: int):
        self.pad_token_id = pad_token_id
        self.max_len      = max_len
        self._printed     = False

    def __call__(self, batch):
        input_ids = [
            torch.tensor(
                list(item["input_ids"][:self.max_len]),  # list() forces copy from numpy int32
                dtype=torch.long,
            )
            for item in batch
        ]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        input_ids_padded = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        if not self._printed:
            print(f"[collate] batch size: {len(batch)}, "
                  f"padded shape: {input_ids_padded.shape}, "
                  f"label dtype: {labels.dtype}", flush=True)
            self._printed = True

        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

def log_config(fold_idx: int, results_fold_dir: Path, extra: dict = None):
    """
    Print and persist the full training configuration for this fold.
    `extra` can carry fold-specific values (e.g. class weights, split sizes).
    """
    config = {**TRAIN_CONFIG, "fold": fold_idx}
    if extra:
        config.update(extra)

    print("\n" + "─" * 60)
    print(f"  CONFIGURATION  (fold {fold_idx})")
    print("─" * 60)
    col_w = max(len(k) for k in config) + 2
    for k, v in config.items():
        print(f"  {k:<{col_w}}: {v}")
    print("─" * 60 + "\n")

    config_path = results_fold_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved → {config_path}", flush=True)
    return config


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    Return the path to the most recent Trainer checkpoint directory, or None.
    Trainer writes checkpoints as checkpoint-<global_step>/.
    """
    ckpt_dirs = sorted(
        glob.glob(str(checkpoint_dir / "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    if ckpt_dirs:
        latest = ckpt_dirs[-1]
        print(f"  Resuming from checkpoint: {latest}", flush=True)
        return latest
    return None


def freeze_base_layers(model, n_layers_to_freeze: int):
    """Freeze embeddings + first n_layers_to_freeze encoder layers."""
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:n_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")


def make_compute_metrics(donor_ids: np.ndarray):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Trainer pads the last batch when dataset size % batch_size != 0.
        # Trim logits/labels back to the true dataset length.
        n = len(donor_ids)
        logits = logits[:n]
        labels = labels[:n]

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=-1)

        auroc = roc_auc_score(labels, probs[:, 1])
        aupr  = average_precision_score(labels, probs[:, 1])
        f1    = f1_score(labels, preds, average="binary")

        df = pd.DataFrame({
            "donor_id": donor_ids,
            "pred":     preds,
            "label":    labels,
        })
        donor_df = df.groupby("donor_id").agg(
            majority_pred=("pred",  lambda x: int(x.mode()[0])),
            true_label   =("label", lambda x: int(x.mode()[0])),
        )
        donor_acc = (donor_df["majority_pred"] == donor_df["true_label"]).mean()

        return {"auroc": auroc, "aupr": aupr, "f1": f1, "donor_acc": donor_acc}
    return compute_metrics


def class_weights_from_labels(labels) -> torch.Tensor:
    """Inverse-frequency class weights to handle PD/healthy imbalance."""
    counts  = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float)


class GPUStatsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved  = torch.cuda.memory_reserved(0) / 1e9
            util      = torch.cuda.utilization(0)
            print(f"  [GPU] mem allocated: {allocated:.1f}GB | "
                  f"mem reserved: {reserved:.1f}GB | "
                  f"utilization: {util}%")

class BestModelCallback(TrainerCallback):
    """Saves the best model by validation AUROC independent of checkpoint cadence."""
    def __init__(self, save_path: Path):
        self.save_path  = save_path
        self.best_auroc = -1.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        auroc = metrics.get("eval_auroc", -1.0)
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            kwargs["model"].save_pretrained(str(self.save_path))
            print(f"  [BestModelCallback] New best AUROC {auroc:.4f} — "
                  f"model saved to {self.save_path}", flush=True)

class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class loss weighting."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weight  = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Main fold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(fold_idx: int, folds: list):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1} / {N_CV_FOLDS}")
    print(f"{'='*60}")

    results_fold_dir = RESULTS_DIR / f"fold_{fold_idx}"
    results_fold_dir.mkdir(exist_ok=True)
    checkpoint_dir   = results_fold_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ── Load train / val / test splits ────────────────────────────────────
    splits_fold_dir = SPLIT_DIR / f"fold_{fold_idx}"
    ds = DatasetDict({
        "train": load_from_disk(str(splits_fold_dir / "train")),
        "val":   load_from_disk(str(splits_fold_dir / "val")),
        "test":  load_from_disk(str(splits_fold_dir / "test")),
    })
    # ── Debug: inspect dataset schema ─────────────────────────────────────────
    print("split folds dir", splits_fold_dir, flush=True)
    print("Train dataset features:", ds["train"].features, flush=True)
    print("First train item keys:", list(ds["train"][0].keys()), flush=True)
    print("First train item:", ds["train"][0], flush=True)

    train_labels  = ds["train"]["label"]
    class_weights = class_weights_from_labels(train_labels)
    cw_healthy, cw_pd = float(class_weights[0]), float(class_weights[1])

    print(f"  Class weights: healthy={cw_healthy:.3f}, PD={cw_pd:.3f}", flush=True)
    print(f"  Train size : {len(ds['train'])}", flush=True)
    print(f"  Val size   : {len(ds['val'])}", flush=True)
    print(f"  Test size  : {len(ds['test'])}", flush=True)
    print(f"  Steps/epoch: {len(ds['train']) // BATCH_SIZE}", flush=True)

    # ── Log & save configuration ───────────────────────────────────────────
    fold_extra = {
        "class_weight_healthy": cw_healthy,
        "class_weight_pd":      cw_pd,
        "n_train_cells":        len(ds["train"]),
        "n_val_cells":          len(ds["val"]),
        "n_test_cells":         len(ds["test"]),
        "steps_per_epoch":      len(ds["train"]) // BATCH_SIZE,
        "train_donors":         folds[fold_idx]["train"],
        "val_donors":           folds[fold_idx].get("val", []),
        "test_donors":          folds[fold_idx]["test"],
        "timestamp":            datetime.now().isoformat(),
    }
    run_config = log_config(fold_idx, results_fold_dir, extra=fold_extra)

    # ── Initialise W&B run ─────────────────────────────────────────────────
    # resume="allow" means W&B will continue the same run if the run_id file
    # exists (written on first launch); otherwise it creates a new run.
    wandb_id_file = results_fold_dir / "wandb_run_id.txt"
    if wandb_id_file.exists():
        wandb_run_id = wandb_id_file.read_text().strip()
        print(f"  Resuming W&B run {wandb_run_id}", flush=True)
    else:
        wandb_run_id = wandb.util.generate_id()
        wandb_id_file.write_text(wandb_run_id)
        print(f"  New W&B run id: {wandb_run_id}", flush=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"fold_{fold_idx}",
        id=wandb_run_id,
        resume="allow",
        config=run_config,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"  Loading {PRETRAINED_MODEL} ...", flush=True)
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}", flush=True)
    model = model.to(device)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU memory after model load: {allocated:.2f} GB", flush=True)
        if allocated < 0.1:
            raise RuntimeError("Model does not appear to be on GPU — check PyTorch CUDA build")

    if FREEZE_LAYERS > 0:
        print(f"  Freezing bottom {FREEZE_LAYERS} layers...", flush=True)
        freeze_base_layers(model, FREEZE_LAYERS)

    # ── Collator ───────────────────────────────────────────────────────────
    with open(f"{GENEFORMER_REPO}/geneformer/token_dictionary_gc104M.pkl", "rb") as f:
        token_dict = pickle.load(f)

    pad_token_id = token_dict.get("<pad>", 0)
    print(f"  Pad token id: {pad_token_id}", flush=True)

    collator = GeneformerCollator(pad_token_id=pad_token_id, max_len=MAX_INPUT_SIZE)
    # ── Detect latest checkpoint for resumption ────────────────────────────
    resume_from = find_latest_checkpoint(checkpoint_dir)

    # ── Training args ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        evaluation_strategy="epoch",      # eval + early stopping stays per-epoch
        save_strategy="steps",            # checkpoints for preemption recovery
        save_steps=1000,                   # ~1 hour on A100 with batch=96; tune as needed
        load_best_model_at_end=False,     # decoupled, so this must be off
        save_total_limit=2,               # keep only the 2 most recent recovery checkpoints
        metric_for_best_model="auroc",
        greater_is_better=True,

        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        fp16=FP16,
        dataloader_num_workers=0,
        logging_steps=50,
        seed=RANDOM_SEED,

        report_to="wandb",
        run_name=f"fold_{fold_idx}",
    )

    val_donor_ids = np.array(ds["val"]["donor_id"])

    best_model_dir = results_fold_dir / "best_model"

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        compute_metrics=make_compute_metrics(val_donor_ids),
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            GPUStatsCallback(),
            BestModelCallback(save_path=best_model_dir),
        ],
        data_collator=collator,
    )

    # ── Train (resume if a checkpoint exists) ─────────────────────────────
    print("  Starting training...", flush=True)
    if resume_from:
        print(f"  ↳ Resuming from {resume_from} "
              f"(epoch tracking and optimizer state preserved)", flush=True)
    print(f"  Truncating sequences to MAX_INPUT_SIZE={MAX_INPUT_SIZE}", flush=True)

    trainer.train(resume_from_checkpoint=resume_from)
    print("Loading best model from directory ", str(best_model_dir))
    model = BertForSequenceClassification.from_pretrained(str(best_model_dir))
    model = model.to(device)
    trainer.model = model
    # ── Validation metrics ─────────────────────────────────────────────────
    print("  Evaluating on validation set (best checkpoint)...", flush=True)
    val_out    = trainer.predict(ds["val"])
    n_val = len(ds["val"])
    val_logits = val_out.predictions[:n_val]
    val_labels = val_out.label_ids[:n_val]
    val_probs  = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_preds  = np.argmax(val_logits, axis=-1)

    val_metrics = {
        "auroc": roc_auc_score(val_labels, val_probs[:, 1]),
        "aupr":  average_precision_score(val_labels, val_probs[:, 1]),
        "f1":    f1_score(val_labels, val_preds, average="binary"),
    }
    print(f"  [Val]  AUROC: {val_metrics['auroc']:.4f}  "
          f"AUPR: {val_metrics['aupr']:.4f}  "
          f"F1: {val_metrics['f1']:.4f}")
    wandb.log({f"val_final/{k}": v for k, v in val_metrics.items()})

    # ── Test evaluation (one-shot) ─────────────────────────────────────────
    print("  Evaluating on held-out test set (one-shot)...", flush=True)
    test_out    = trainer.predict(ds["test"])
    n_test = len(ds["test"])
    test_logits = test_out.predictions[:n_test]
    test_labels = test_out.label_ids[:n_test]
    test_probs  = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_preds  = np.argmax(test_logits, axis=-1)

    fold_metrics = {
        "fold":         fold_idx,
        "auroc":        roc_auc_score(test_labels, test_probs[:, 1]),
        "aupr":         average_precision_score(test_labels, test_probs[:, 1]),
        "f1":           f1_score(test_labels, test_preds, average="binary"),
        "val_auroc":    val_metrics["auroc"],
        "val_aupr":     val_metrics["aupr"],
        "val_f1":       val_metrics["f1"],
        "n_test_cells": len(test_labels),
        "test_donors":  folds[fold_idx]["test"],
    }
    print(f"  [Test] AUROC: {fold_metrics['auroc']:.4f}  "
          f"AUPR: {fold_metrics['aupr']:.4f}  "
          f"F1: {fold_metrics['f1']:.4f}")
    print("\n" + classification_report(test_labels, test_preds,
                                       target_names=["healthy", "Parkinson"]))

    wandb.log({
        "test/auroc": fold_metrics["auroc"],
        "test/aupr":  fold_metrics["aupr"],
        "test/f1":    fold_metrics["f1"],
    })
    wandb.finish()

    # ── Persist outputs ────────────────────────────────────────────────────
    pd.DataFrame({
        "label":    test_labels,
        "pred":     test_preds,
        "prob_pd":  test_probs[:, 1],
        "donor_id": ds["test"]["donor_id"],
    }).to_csv(results_fold_dir / "test_predictions.csv", index=False)

    pd.DataFrame({
        "label":    val_labels,
        "pred":     val_preds,
        "prob_pd":  val_probs[:, 1],
        "donor_id": ds["val"]["donor_id"],
    }).to_csv(results_fold_dir / "val_predictions.csv", index=False)

    trainer.save_model(str(results_fold_dir / "best_model"))

    with open(results_fold_dir / "metrics.pkl", "wb") as f:
        pickle.dump(fold_metrics, f)

    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold (0-indexed). Omit to run all.")
    args = parser.parse_args()

    with open(SPLIT_DIR / "cv_folds.pkl", "rb") as f:
        folds = pickle.load(f)

    fold_range  = [args.fold] if args.fold is not None else range(N_CV_FOLDS)
    all_metrics = []

    for fold_idx in fold_range:
        m = run_fold(fold_idx, folds)
        all_metrics.append(m)

    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        metrics_df = pd.DataFrame(all_metrics)
        for split, prefix in [("Test", ""), ("Val", "val_")]:
            print(f"\n  {split} metrics:")
            for metric in ["auroc", "aupr", "f1"]:
                col  = f"{prefix}{metric}"
                vals = metrics_df[col]
                print(f"    {metric.upper()}: {vals.mean():.4f} ± {vals.std():.4f}")
        metrics_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
        print(f"\nFull results saved to {RESULTS_DIR}/cv_metrics.csv")


if __name__ == "__main__":
    main()