"""
Step 2 (smoke test): Quick sanity check of the fine-tuning pipeline.

Loads fold 0 from the pre-split datasets (output of 01b_split_folds.py),
subsamples a small number of cells, and runs for 1 epoch. If this completes
without error and prints a loss + eval metrics, the full run is ready to go.

Usage:
    python 02_smoketest.py

Expected runtime: ~2-5 minutes on a GPU.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
# ── Geneformer imports ────────────────────────────────────────────────────────
GENEFORMER_REPO = os.environ.get("GENEFORMER_REPO", "./Geneformer")
sys.path.insert(0, GENEFORMER_REPO)
from geneformer import TranscriptomeTokenizer  # noqa: E402

DONOR_COL    = "donor_id"

# ── Config ────────────────────────────────────────────────────────────────────
GENEFORMER_REPO  = "/orcd/compute/edsun/001/jholz/finetuning/Geneformer"
OUT_DIR   = Path("/orcd/compute/edsun/001/jholz/finetuning/inputs")
SPLIT_DIR        = OUT_DIR / "splits"
SMOKE_RESULTS    = OUT_DIR / "smoketest_results"
SMOKE_RESULTS.mkdir(parents=True, exist_ok=True)

PRETRAINED_MODEL = f"{GENEFORMER_REPO}/Geneformer-V2-104M"
FOLD_IDX         = 0        # always use fold 0 for the smoke test
N_TRAIN_CELLS    = 1000     # cells to subsample for training
N_TEST_CELLS     = 200      # cells to subsample for evaluation
BATCH_SIZE       = 8
NUM_LABELS       = 2
RANDOM_SEED      = 42
FP16             = torch.cuda.is_available()

N_PROC = 8
sys.path.insert(0, GENEFORMER_REPO)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load pre-split fold 0 datasets
# ─────────────────────────────────────────────────────────────────────────────
fold_dir = SPLIT_DIR / f"fold_{FOLD_IDX}"
print(f"Loading fold {FOLD_IDX} from {fold_dir}...")

train_ds = load_from_disk(str(fold_dir / "train"))
test_ds  = load_from_disk(str(fold_dir / "test"))

print(f"  Full train: {len(train_ds):,} cells")
print(f"  Full test:  {len(test_ds):,} cells")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Subsample — stratified by label so both classes are present
# ─────────────────────────────────────────────────────────────────────────────
def stratified_subsample(dataset, n, seed=RANDOM_SEED):
    """Return n cells sampled with equal class balance."""
    labels   = np.array(dataset["label"])
    n_each   = n // 2
    rng      = np.random.default_rng(seed)
    idx_0    = rng.choice(np.where(labels == 0)[0], size=min(n_each, (labels==0).sum()), replace=False)
    idx_1    = rng.choice(np.where(labels == 1)[0], size=min(n_each, (labels==1).sum()), replace=False)
    idx      = np.concatenate([idx_0, idx_1])
    rng.shuffle(idx)
    return dataset.select(idx.tolist())

train_small = stratified_subsample(train_ds, N_TRAIN_CELLS)
test_small  = stratified_subsample(test_ds,  N_TEST_CELLS)

train_labels = train_small["label"]
test_labels  = test_small["label"]
print(f"\nSmoke test subset:")
print(f"  Train: {len(train_small)} cells  "
      f"(PD={sum(train_labels)}, healthy={sum(1-l for l in train_labels)})")
print(f"  Test:  {len(test_small)} cells   "
      f"(PD={sum(test_labels)}, healthy={sum(1-l for l in test_labels)})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Class weights
# ─────────────────────────────────────────────────────────────────────────────
def class_weights_from_labels(labels):
    counts  = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float)

class_weights = class_weights_from_labels(train_labels)
print(f"  Class weights: healthy={class_weights[0]:.3f}, PD={class_weights[1]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load model
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading model from {PRETRAINED_MODEL}...")
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {trainable:,} trainable / {total:,} total")
print(f"  GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Weighted trainer
# ─────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weight  = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss    = torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds  = np.argmax(logits, axis=-1)
    # Guard against single-class edge case in tiny subsample
    if len(np.unique(labels)) < 2:
        print("  WARNING: only one class in eval set — AUROC undefined, returning 0.5")
        return {"auroc": 0.5, "aupr": 0.5, "f1": 0.0}
    return {
        "auroc": roc_auc_score(labels, probs[:, 1]),
        "aupr":  average_precision_score(labels, probs[:, 1]),
        "f1":    f1_score(labels, preds, average="binary"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. Training args — minimal, 1 epoch only
# ─────────────────────────────────────────────────────────────────────────────


# Load the pad token id directly from Geneformer's token dictionary
with open(f"{GENEFORMER_REPO}/geneformer/token_dictionary_gc104M.pkl", "rb") as f:
    token_dict = pickle.load(f)

pad_token_id = token_dict.get("<pad>", 0)
print("token dict", token_dict, flush=True)
print(f"  Pad token id: {pad_token_id}", flush=True)

# Custom collator that pads input_ids to the longest sequence in the batch
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels    = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Pad to longest in batch
    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id,
    )

    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids":      input_ids_padded,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

training_args = TrainingArguments(
    output_dir=str(SMOKE_RESULTS / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="no",           # don't save checkpoints in smoke test
    logging_steps=10,
    fp16=FP16,
    dataloader_num_workers=0,
    seed=RANDOM_SEED,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_small,
    eval_dataset=test_small,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    data_collator=collate_fn,  # add this line
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\nStarting smoke test training (1 epoch)...")
train_result = trainer.train()

print("\nEvaluating...")
metrics = trainer.evaluate()

print("\n" + "="*50)
print("SMOKE TEST COMPLETE")
print("="*50)
print(f"  Train loss:  {train_result.training_loss:.4f}")
print(f"  Eval AUROC:  {metrics.get('eval_auroc', 'N/A')}")
print(f"  Eval AUPR:   {metrics.get('eval_aupr', 'N/A')}")
print(f"  Eval F1:     {metrics.get('eval_f1', 'N/A')}")
print(f"  Runtime:     {train_result.metrics['train_runtime']:.1f}s")
print("\nAll checks passed — safe to launch full fine-tuning with 02_finetune.py")