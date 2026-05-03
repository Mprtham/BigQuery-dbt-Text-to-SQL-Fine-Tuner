"""
Phase 2: Direct Preference Optimisation with TRL DPOTrainer.
Loads SFT checkpoint and trains on dpo_pairs.jsonl.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "dataset"
SFT_DIR    = ROOT / "models" / "sft-phi3-v1"
OUTPUT_DIR = ROOT / "models" / "dpo-phi3-v1"

# ── Model ─────────────────────────────────────────────────────────────────────

print(f"Loading SFT model from {SFT_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(str(SFT_DIR))
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(SFT_DIR),
    torch_dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map   = "auto",
)

# Reference model is the SFT checkpoint (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    str(SFT_DIR),
    torch_dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map   = "auto",
)

# ── Dataset ───────────────────────────────────────────────────────────────────

dpo_dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR / "dpo_pairs.jsonl"),
    split="train",
)

# DPOTrainer expects columns: prompt, chosen, rejected
dpo_dataset = dpo_dataset.select_columns(["prompt", "chosen", "rejected"])

split = dpo_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"DPO train: {len(train_dataset)} | eval: {len(eval_dataset)}")

# ── Training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = 1,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 4,
    warmup_ratio                = 0.1,
    learning_rate               = 5e-5,
    lr_scheduler_type           = "cosine",
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 5,
    evaluation_strategy         = "steps",
    eval_steps                  = 20,
    save_strategy               = "steps",
    save_steps                  = 50,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    report_to                   = "none",
    seed                        = 42,
    remove_unused_columns       = False,
)

# ── DPO Trainer ───────────────────────────────────────────────────────────────

dpo_trainer = DPOTrainer(
    model            = model,
    ref_model        = ref_model,
    tokenizer        = tokenizer,
    args             = training_args,
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    beta             = 0.1,           # KL penalty weight
    max_length       = 1024,
    max_prompt_length= 512,
)

# ── Run ───────────────────────────────────────────────────────────────────────

print("Starting DPO training...")
dpo_trainer.train()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
dpo_trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"DPO model saved to {OUTPUT_DIR}")
