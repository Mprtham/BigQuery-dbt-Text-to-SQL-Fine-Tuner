"""
Generate DPO preference pairs: (prompt, chosen_good_sql, rejected_bad_sql).
Rejection types match the design doc. Outputs data/dataset/dpo_pairs.jsonl.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

from groq import Groq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from synthetic.bq_templates import ALL_TEMPLATES, SQLTemplate
from synthetic.dbt_templates import DBT_TEMPLATES, DBTTemplate

GROQ_MODEL         = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"
OUTPUT_DIR = ROOT / "data" / "dataset"
TARGET_DPO_PAIRS = 200

# ── Rejection mutation functions ──────────────────────────────────────────────

def mutate_mysql_dialect(sql: str) -> tuple[str, str]:
    """Replace BQ date functions with MySQL equivalents."""
    rejected = sql
    rejected = rejected.replace("FORMAT_DATE('%Y-%m', ", "DATE_FORMAT(, '%Y-%m'")
    rejected = rejected.replace("TIMESTAMP_SUB(", "DATE_SUB(")
    rejected = rejected.replace("TIMESTAMP_ADD(", "DATE_ADD(")
    rejected = rejected.replace("DATE_TRUNC(", "DATE_FORMAT(")
    rejected = rejected.replace("EXTRACT(YEAR FROM ", "YEAR(")
    if rejected == sql:
        return None, None
    return sql, rejected


def mutate_no_qualify(sql: str) -> tuple[str, str]:
    """Replace QUALIFY with a nested subquery (verbose anti-pattern)."""
    if "QUALIFY" not in sql:
        return None, None
    chosen = sql
    rejected = sql.replace(
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1",
        ""
    )
    rejected = f"""SELECT * FROM (
  {rejected.strip()}
  ,ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) AS _rn
) WHERE _rn = 1"""
    return chosen, rejected


def mutate_wrong_date_function(sql: str) -> tuple[str, str]:
    """Swap FORMAT_DATE for FORMAT_TIMESTAMP (wrong for DATE type)."""
    if "FORMAT_DATE" not in sql:
        return None, None
    return sql, sql.replace("FORMAT_DATE(", "FORMAT_TIMESTAMP(")


def mutate_missing_unnest(sql: str) -> tuple[str, str]:
    """Remove UNNEST — produces a syntax error on array column access."""
    if "UNNEST" not in sql:
        return None, None
    chosen = sql
    rejected = sql.replace(", UNNEST(items) AS item", "")
    rejected = rejected.replace("item.sku", "items.sku")
    rejected = rejected.replace("item.qty", "items.qty")
    rejected = rejected.replace("item.unit_price", "items.unit_price")
    return chosen, rejected


def mutate_select_star(sql: str) -> tuple[str, str]:
    """Replace explicit column list with SELECT * (bad practice)."""
    lines = sql.strip().splitlines()
    if not lines:
        return None, None
    if lines[0].strip().upper().startswith("SELECT\n") or lines[0].strip().upper() == "SELECT":
        return None, None
    chosen = sql
    select_end = sql.find("\nFROM ")
    if select_end == -1:
        return None, None
    rejected = "SELECT *" + sql[select_end:]
    return chosen, rejected


def mutate_wrong_jinja(sql: str) -> tuple[str, str]:
    """Replace {% if is_incremental() %} with {% if this %} (wrong usage)."""
    if "is_incremental()" not in sql:
        return None, None
    return sql, sql.replace("is_incremental()", "this")


MUTATORS = [
    ("mysql_dialect",       mutate_mysql_dialect),
    ("no_qualify",          mutate_no_qualify),
    ("wrong_date_function", mutate_wrong_date_function),
    ("missing_unnest",      mutate_missing_unnest),
    ("select_star",         mutate_select_star),
    ("wrong_jinja",         mutate_wrong_jinja),
]

# ── LLM-assisted pair generation ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert BigQuery SQL engineer generating training data for DPO fine-tuning.
Given a correct BigQuery SQL query, generate a WRONG version that exhibits a specific anti-pattern.

Output exactly two blocks:
### CHOSEN:
<the correct SQL>

### REJECTED:
<the wrong SQL with the requested anti-pattern>

Anti-pattern types and what to do:
- mysql_dialect: Use DATE_FORMAT instead of FORMAT_DATE, DATE_SUB instead of TIMESTAMP_SUB, YEAR() instead of EXTRACT(YEAR FROM ...)
- no_qualify: Replace QUALIFY ROW_NUMBER()=1 with a nested subquery pattern
- wrong_date_function: Use FORMAT_TIMESTAMP where FORMAT_DATE is correct
- missing_unnest: Remove UNNEST and access array fields with dot notation (causes syntax error)
- select_star: Replace the explicit SELECT column list with SELECT *
- wrong_jinja: Use {% if this %} instead of {% if is_incremental() %}"""


def _call_groq(client: Groq, model: str, user_msg: str) -> str | None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def generate_pair_with_llm(
    client: Groq, prompt: str, sql: str, anti_pattern: str
) -> tuple[str, str] | None:
    user_msg = f"""Anti-pattern to introduce: {anti_pattern}

Correct SQL:
{sql}

Generate the wrong version following the anti-pattern rules."""
    for model in (GROQ_MODEL, GROQ_MODEL_FALLBACK):
        try:
            raw = _call_groq(client, model, user_msg)
            if "### CHOSEN:" not in raw or "### REJECTED:" not in raw:
                return None
            chosen   = raw.split("### CHOSEN:")[1].split("### REJECTED:")[0].strip()
            rejected = raw.split("### REJECTED:")[1].strip()
            if len(chosen) < 20 or len(rejected) < 20:
                return None
            return chosen, rejected
        except Exception as e:
            err_str = str(e)
            if "rate_limit_exceeded" in err_str and "tokens per day" in err_str:
                print(f"    [WARN] Daily token limit on {model}, trying fallback...")
                continue
            print(f"    [WARN] Groq error ({model}): {e}")
            return None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(dry_run: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []

    all_templates = [(t.question, t.sql) for t in ALL_TEMPLATES + DBT_TEMPLATES]
    random.shuffle(all_templates)

    # First pass: deterministic mutations (no API needed)
    for question, sql in all_templates:
        for pattern_name, mutator in MUTATORS:
            chosen, rejected = mutator(sql)
            if chosen and rejected and chosen != rejected:
                prompt = f"### Question:\n{question}\n\n### SQL:"
                pairs.append({
                    "prompt":         prompt,
                    "chosen":         "\n" + chosen,
                    "rejected":       "\n" + rejected,
                    "rejection_type": pattern_name,
                })
        if len(pairs) >= TARGET_DPO_PAIRS:
            break

    print(f"Deterministic mutations: {len(pairs)} pairs")

    # Second pass: LLM-assisted mutations to fill quota (cycle through templates)
    if not dry_run and len(pairs) < TARGET_DPO_PAIRS:
        import itertools
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        anti_patterns = [name for name, _ in MUTATORS]
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10

        for question, sql in itertools.cycle(all_templates):
            if len(pairs) >= TARGET_DPO_PAIRS:
                break
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  [INFO] {consecutive_failures} consecutive failures — saving partial results.")
                break
            pattern = random.choice(anti_patterns)
            prompt = f"### Question:\n{question}\n\n### SQL:"
            print(f"  LLM pair [{pattern}] — have {len(pairs)}/{TARGET_DPO_PAIRS}")
            result = generate_pair_with_llm(client, prompt, sql, pattern)
            if result:
                chosen, rejected = result
                pairs.append({
                    "prompt":         prompt,
                    "chosen":         "\n" + chosen,
                    "rejected":       "\n" + rejected,
                    "rejection_type": pattern,
                })
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            time.sleep(0.3)

    random.shuffle(pairs)
    out_path = OUTPUT_DIR / "dpo_pairs.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone. {len(pairs)} DPO pairs saved to {out_path}")
    counts = {}
    for p in pairs:
        counts[p["rejection_type"]] = counts.get(p["rejection_type"], 0) + 1
    print("Rejection type distribution:", counts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate(dry_run=args.dry_run)
