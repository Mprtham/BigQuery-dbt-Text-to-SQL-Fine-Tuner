"""
Generate SFT training examples using Groq (llama-3.1-70b).
Seeds generation from bq_templates and dbt_templates, then asks the LLM
to produce N variations per template. Outputs data/dataset/sft_train.jsonl.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import yaml
from groq import Groq

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from synthetic.bq_templates import ALL_TEMPLATES, SQLTemplate
from synthetic.dbt_templates import DBT_TEMPLATES, DBTTemplate

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
OUTPUT_DIR = ROOT / "data" / "dataset"
SCHEMA_DIR = ROOT / "data" / "schemas"

TARGET_COUNTS = {
    "easy":   150,
    "medium": 250,
    "hard":   100,
    "dbt":    100,
}

TOTAL_TARGET = sum(TARGET_COUNTS.values())  # 600

# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert BigQuery SQL and dbt engineer.
Your task is to generate high-quality Text-to-SQL training examples.
Each example must follow the exact format below — no deviations.

Rules:
- Use BigQuery-native SQL syntax ONLY (not MySQL, PostgreSQL, or generic SQL)
- Use backtick-quoted fully-qualified table names: `project.dataset.table`
- Prefer QUALIFY over nested ROW_NUMBER subqueries for deduplication
- Use FORMAT_DATE, DATE_TRUNC, TIMESTAMP_SUB — never DATE_FORMAT or DATEADD
- Use COUNTIF, ARRAY_AGG, UNNEST, STRUCT where appropriate
- Do not use SELECT * in any generated SQL
- For dbt examples: use {{ ref() }}, {{ source() }}, {% if is_incremental() %} correctly

Output format (exactly this, nothing else):
### Schema:
<schema block>

### Question:
<natural language question>

### SQL:
<BigQuery SQL or dbt Jinja SQL>"""


def load_schema(table_name: str) -> str:
    schema_file = SCHEMA_DIR / f"{table_name}.yaml"
    if not schema_file.exists():
        return f"table: {table_name}\ncolumns: [unknown]"
    with open(schema_file) as f:
        data = yaml.safe_load(f)
    lines = [f"table: {data['table']}"]
    lines.append("columns:")
    for col in data.get("columns", []):
        line = f"  - name: {col['name']}  type: {col['type']}"
        if col.get("description"):
            line += f"  description: {col['description']}"
        if col.get("values"):
            line += f"  values: {col['values']}"
        lines.append(line)
    return "\n".join(lines)


def make_variation_prompt(template: SQLTemplate | DBTTemplate, n: int) -> str:
    if isinstance(template, SQLTemplate):
        tables = [t.strip() for t in template.schema_table.split(",")]
        schema_block = "\n\n".join(load_schema(t) for t in tables)
    else:
        schema_block = "# dbt model — uses ref() / source() — schema from upstream models"

    return f"""Here is one example of a {template.tier}-tier BigQuery SQL training example:

--- EXAMPLE START ---
### Schema:
{schema_block}

### Question:
{template.question}

### SQL:
{template.sql}
--- EXAMPLE END ---

Now generate {n} NEW and DIFFERENT training examples at the same difficulty tier ({template.tier}).
Each example must follow the exact ### Schema / ### Question / ### SQL format.
Vary the tables, questions, SQL patterns, and column names.
Separate each example with a line containing only: ---"""


# ── Parsing LLM output ────────────────────────────────────────────────────────

def parse_llm_output(text: str) -> list[dict]:
    examples = []
    blocks = text.split("---")
    for block in blocks:
        block = block.strip()
        if "### Schema:" not in block or "### Question:" not in block or "### SQL:" not in block:
            continue
        try:
            schema_part  = block.split("### Question:")[0].replace("### Schema:", "").strip()
            question_sql = block.split("### Question:")[1]
            question     = question_sql.split("### SQL:")[0].strip()
            sql          = question_sql.split("### SQL:")[1].strip()
            if len(sql) < 20 or len(question) < 10:
                continue
            examples.append({
                "schema":   schema_part,
                "question": question,
                "sql":      sql,
            })
        except (IndexError, ValueError):
            continue
    return examples


def format_as_training_example(ex: dict) -> dict:
    prompt = f"### Schema:\n{ex['schema']}\n\n### Question:\n{ex['question']}\n\n### SQL:"
    return {
        "prompt":     prompt,
        "completion": "\n" + ex["sql"],
        "text":       prompt + "\n" + ex["sql"],
    }


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(dry_run: bool = False) -> None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict] = []
    tier_counts: dict[str, int] = {t: 0 for t in TARGET_COUNTS}

    # Seed examples from templates directly (no LLM needed)
    for tmpl in ALL_TEMPLATES:
        tables = [t.strip() for t in tmpl.schema_table.split(",")]
        schema_block = "\n\n".join(load_schema(t) for t in tables)
        ex = {
            "schema":   schema_block,
            "question": tmpl.question,
            "sql":      tmpl.sql,
        }
        all_examples.append(format_as_training_example(ex))
        tier_counts[tmpl.tier] += 1

    for tmpl in DBT_TEMPLATES:
        ex = {
            "schema":   "# dbt model",
            "question": tmpl.question,
            "sql":      tmpl.sql,
        }
        all_examples.append(format_as_training_example(ex))
        tier_counts["dbt"] += 1

    print(f"Seeded {len(all_examples)} examples from templates: {tier_counts}")

    if dry_run:
        print("Dry run — skipping LLM calls.")
    else:
        templates_to_expand = list(ALL_TEMPLATES) + list(DBT_TEMPLATES)
        random.shuffle(templates_to_expand)

        for tmpl in templates_to_expand:
            tier = tmpl.tier
            needed = TARGET_COUNTS[tier] - tier_counts[tier]
            if needed <= 0:
                continue

            n_per_call = min(5, needed)
            prompt = make_variation_prompt(tmpl, n_per_call)

            print(f"  [{tier}] Requesting {n_per_call} variations (have {tier_counts[tier]}/{TARGET_COUNTS[tier]})...")
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.8,
                    max_tokens=4096,
                )
                raw = response.choices[0].message.content
                parsed = parse_llm_output(raw)
                for ex in parsed:
                    if tier_counts[tier] >= TARGET_COUNTS[tier]:
                        break
                    all_examples.append(format_as_training_example(ex))
                    tier_counts[tier] += 1
                print(f"    → parsed {len(parsed)} examples. Running totals: {tier_counts}")
            except Exception as e:
                print(f"    [WARN] Groq error: {e}")
            time.sleep(0.5)

    # Shuffle and split 90/10 train/test
    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.9)
    train = all_examples[:split]
    test  = all_examples[split:]

    train_path = OUTPUT_DIR / "sft_train.jsonl"
    test_path  = OUTPUT_DIR / "sft_test.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone. Train: {len(train)} | Test: {len(test)}")
    print(f"Saved to {train_path} and {test_path}")
    print(f"Final tier counts: {tier_counts}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Seed from templates only, skip LLM calls")
    args = parser.parse_args()
    generate(dry_run=args.dry_run)
