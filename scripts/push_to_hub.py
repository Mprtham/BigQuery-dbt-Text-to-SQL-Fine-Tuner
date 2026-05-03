"""
Push trained model and datasets to HuggingFace Hub.
Usage:
  python scripts/push_to_hub.py --push-model --push-dataset
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "dataset"
MODEL_DIR  = ROOT / "models" / "dpo-phi3-v1"
SFT_DIR    = ROOT / "models" / "sft-phi3-v1"

HF_USERNAME   = "Mpratham"
SFT_REPO      = f"{HF_USERNAME}/bq-dbt-sql-sft"
DPO_REPO      = f"{HF_USERNAME}/bq-dbt-sql-dpo"
MODEL_REPO    = f"{HF_USERNAME}/sql-forge-phi3-v1"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def push_sft_dataset() -> None:
    print(f"Pushing SFT dataset to {SFT_REPO}...")
    train = load_jsonl(DATA_DIR / "sft_train.jsonl")
    test  = load_jsonl(DATA_DIR / "sft_test.jsonl")

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "test":  Dataset.from_list(test),
    })
    ds.push_to_hub(SFT_REPO, private=False)
    print(f"SFT dataset pushed: {len(train)} train / {len(test)} test")


def push_dpo_dataset() -> None:
    print(f"Pushing DPO dataset to {DPO_REPO}...")
    pairs = load_jsonl(DATA_DIR / "dpo_pairs.jsonl")
    ds = Dataset.from_list(pairs)
    ds.push_to_hub(DPO_REPO, private=False)
    print(f"DPO dataset pushed: {len(pairs)} pairs")


def push_model() -> None:
    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else str(SFT_DIR)
    print(f"Pushing model from {model_path} to {MODEL_REPO}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.push_to_hub(MODEL_REPO, private=False)
    tokenizer.push_to_hub(MODEL_REPO, private=False)
    print(f"Model pushed to {MODEL_REPO}")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN environment variable before pushing.")

    # Login once so all library calls (datasets + transformers) pick up the token
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--push-model",   action="store_true")
    parser.add_argument("--push-dataset", action="store_true")
    args = parser.parse_args()

    if not args.push_model and not args.push_dataset:
        parser.error("Specify at least one of --push-model or --push-dataset")

    if args.push_dataset:
        push_sft_dataset()
        push_dpo_dataset()

    if args.push_model:
        push_model()
