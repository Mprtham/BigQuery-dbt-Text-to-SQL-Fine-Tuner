"""
Validate SFT and DPO dataset files before training.
Checks: required fields, non-empty values, SQL parse validity (sqlglot BQ dialect).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import bq_parse_valid

DATA_DIR = ROOT / "data" / "dataset"

SFT_REQUIRED  = {"prompt", "completion", "text"}
DPO_REQUIRED  = {"prompt", "chosen", "rejected"}


def validate_file(path: Path, required_keys: set, check_sql: bool = True) -> dict:
    if not path.exists():
        return {"error": f"File not found: {path}", "valid": 0, "invalid": 0, "total": 0}

    valid = invalid = 0
    parse_valid = parse_invalid = 0
    errors = []

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON parse error: {e}")
                invalid += 1
                continue

            missing = required_keys - set(row.keys())
            if missing:
                errors.append(f"Line {i}: missing keys {missing}")
                invalid += 1
                continue

            empty = [k for k in required_keys if not str(row.get(k, "")).strip()]
            if empty:
                errors.append(f"Line {i}: empty values for {empty}")
                invalid += 1
                continue

            if check_sql:
                sql_field = "completion" if "completion" in row else "chosen"
                ok, err = bq_parse_valid(row[sql_field])
                if ok:
                    parse_valid += 1
                else:
                    parse_invalid += 1

            valid += 1

    return {
        "path":          str(path),
        "total":         valid + invalid,
        "valid":         valid,
        "invalid":       invalid,
        "parse_valid":   parse_valid,
        "parse_invalid": parse_invalid,
        "errors":        errors[:20],  # cap at 20
    }


def main():
    files = {
        "SFT train":  (DATA_DIR / "sft_train.jsonl", SFT_REQUIRED),
        "SFT test":   (DATA_DIR / "sft_test.jsonl",  SFT_REQUIRED),
        "DPO pairs":  (DATA_DIR / "dpo_pairs.jsonl", DPO_REQUIRED),
    }

    all_ok = True
    for label, (path, keys) in files.items():
        result = validate_file(path, keys)
        status = "OK" if result["invalid"] == 0 else "!!"
        print(f"\n{status} {label}: {result['total']} total | {result['valid']} valid | {result['invalid']} invalid")
        if result.get("parse_valid") is not None:
            pv = result["parse_valid"]
            pi = result["parse_invalid"]
            print(f"  SQL parse: {pv} valid / {pi} invalid")
        if result["errors"]:
            all_ok = False
            for err in result["errors"]:
                print(f"  ERROR: {err}")

    print()
    if all_ok:
        print("All dataset files passed validation.")
    else:
        print("Validation failed — fix errors before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
