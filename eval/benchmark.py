"""
Benchmark: compare base Phi-3, SFT, and DPO model on eval/gold_set.jsonl.
Usage: python eval/benchmark.py [--models base sft dpo]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import evaluate_batch, print_metrics

GOLD_SET  = Path(__file__).parent / "gold_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_PATHS = {
    "base": "microsoft/Phi-3-mini-4k-instruct",
    "sft":  str(ROOT / "models" / "sft-phi3-v1"),
    "dpo":  str(ROOT / "models" / "dpo-phi3-v1"),
}


def load_gold_set() -> list[dict]:
    examples = []
    with open(GOLD_SET) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_prompt(example: dict) -> str:
    return f"### Schema:\n{example['schema']}\n\n### Question:\n{example['question']}\n\n### SQL:"


def run_inference(model_key: str, examples: list[dict]) -> list[str]:
    model_path = MODEL_PATHS[model_key]
    print(f"\nLoading {model_key} model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map  = "auto",
    )

    gen_pipeline = pipeline(
        "text-generation",
        model        = model,
        tokenizer    = tokenizer,
        max_new_tokens = 512,
        do_sample    = False,
        temperature  = 1.0,
        pad_token_id = tokenizer.eos_token_id,
    )

    predictions = []
    for ex in examples:
        prompt = build_prompt(ex)
        output = gen_pipeline(prompt)[0]["generated_text"]
        sql = output[len(prompt):].strip()
        # Trim at first empty line to avoid hallucinated text after SQL
        sql = sql.split("\n\n")[0].strip()
        predictions.append(sql)

    del model
    torch.cuda.empty_cache()
    return predictions


def benchmark(model_keys: list[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_gold_set()
    print(f"Loaded {len(examples)} gold examples from {GOLD_SET}")

    gold_sqls = [ex["sql"] for ex in examples]
    schema_columns_list = []
    for ex in examples:
        cols = [line.split("name:")[1].split()[0]
                for line in ex["schema"].splitlines()
                if "name:" in line]
        schema_columns_list.append(cols)

    all_results = {}
    for model_key in model_keys:
        predictions = run_inference(model_key, examples)

        results = evaluate_batch(predictions, gold_sqls, schema_columns_list)
        all_results[model_key] = results

        result_path = RESULTS_DIR / f"{model_key}_phi3.json"
        with open(result_path, "w") as f:
            json.dump({"results": results, "predictions": predictions}, f, indent=2)
        print(f"\nResults saved to {result_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    header = f"{'Metric':<28}" + "".join(f"{k:>10}" for k in model_keys)
    print(header)
    print("-" * 60)
    metrics = ["exact_match", "partial_match_avg", "parse_accuracy", "schema_compliance"]
    for metric in metrics:
        row = f"{metric:<28}"
        for k in model_keys:
            val = all_results.get(k, {}).get(metric)
            row += f"{val:>10.1%}" if val is not None else f"{'N/A':>10}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["base", "sft", "dpo"],
        choices=["base", "sft", "dpo"],
    )
    args = parser.parse_args()
    benchmark(args.models)
