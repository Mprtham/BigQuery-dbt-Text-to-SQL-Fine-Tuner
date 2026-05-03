"""
Phase 1: Supervised Fine-Tuning with Unsloth + LoRA.
Trains microsoft/Phi-3-mini-4k-instruct on bq-dbt-sql-sft dataset.
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "dataset"
OUTPUT_DIR = ROOT / "models" / "sft-phi3-v1"

# ── Model config ──────────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 2048
DTYPE          = None   # auto-detect (bfloat16 on Ampere+)
LOAD_IN_4BIT   = False  # 24GB VRAM → full LoRA, no quantisation

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "microsoft/Phi-3-mini-4k-instruct",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = DTYPE,
    load_in_4bit   = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 64,
    lora_alpha     = 128,
    lora_dropout   = 0.0,
    bias           = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)

# ── Dataset ───────────────────────────────────────────────────────────────────

train_dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR / "sft_train.jsonl"),
    split="train",
)
eval_dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR / "sft_test.jsonl"),
    split="train",
)

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = 3,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,          # effective batch = 16
    warmup_ratio                = 0.05,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    evaluation_strategy         = "steps",
    eval_steps                  = 50,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    seed                        = 42,
)

# ── Trainer ───────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    args            = training_args,
)

# ── Run ───────────────────────────────────────────────────────────────────────

print("Starting SFT training...")
trainer_stats = trainer.train()
print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

# Save final model
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Model saved to {OUTPUT_DIR}")

# Save LoRA adapters only (smaller, for DPO phase)
lora_dir = OUTPUT_DIR / "lora_adapters"
model.save_pretrained_merged(str(lora_dir), tokenizer, save_method="lora")
print(f"LoRA adapters saved to {lora_dir}")
