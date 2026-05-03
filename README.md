# sql-forge — BigQuery + dbt SQL Fine-Tuner

Fine-tune `microsoft/Phi-3-mini-4k-instruct` on BigQuery-dialect SQL and dbt Jinja SQL generation, then publish model and dataset to HuggingFace.

**The problem:** No public model natively supports BigQuery syntax (QUALIFY, UNNEST, ARRAY_AGG, STRUCT) combined with dbt macros (`{{ ref() }}`, `{% if is_incremental() %}`).

**Model:** [Mpratham/sql-forge-phi3-v1](https://huggingface.co/Mpratham/sql-forge-phi3-v1)  
**SFT Dataset:** [Mpratham/bq-dbt-sql-sft](https://huggingface.co/datasets/Mpratham/bq-dbt-sql-sft)  
**DPO Dataset:** [Mpratham/bq-dbt-sql-dpo](https://huggingface.co/datasets/Mpratham/bq-dbt-sql-dpo)

---

## Architecture

```
data/schemas/        10 realistic BigQuery table schemas (YAML)
synthetic/           Dataset generation scripts (Groq llama-3.1-70b)
train/               SFT (Unsloth LoRA) + DPO (TRL) training scripts
eval/                Metrics, benchmark, gold set (50 held-out examples)
inference/           FastAPI server + Streamlit playground
scripts/             Dataset validation + HuggingFace publishing
configs/             Hyperparameter YAML configs
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# edit .env — add GROQ_API_KEY and HF_TOKEN

# 3. Generate datasets (dry-run first to verify, no API cost)
make generate-dry
make generate        # calls Groq: 600 SFT + 200 DPO examples

# 4. Validate
make validate

# 5. Train (requires 24GB VRAM — RTX 3090 / 4090)
make train-sft       # Phase 1: 3 epochs SFT
make train-dpo       # Phase 2: 1 epoch DPO

# 6. Benchmark
make benchmark

# 7. Publish
make push
```

---

## Dataset

| Split | Count | Format |
|-------|-------|--------|
| SFT easy | 150 | `{prompt, completion, text}` |
| SFT medium | 250 | `{prompt, completion, text}` |
| SFT hard | 100 | `{prompt, completion, text}` |
| SFT dbt | 100 | `{prompt, completion, text}` |
| DPO pairs | 200 | `{prompt, chosen, rejected, rejection_type}` |

**Prompt format:**
```
### Schema:
table: orders
columns:
  - name: order_id  type: STRING ...

### Question:
For each customer, return only their most recent order.

### SQL:
SELECT order_id, customer_id, created_at
FROM `project.ecommerce.orders`
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1
```

**DPO rejection types:** `mysql_dialect`, `no_qualify`, `wrong_date_function`, `missing_unnest`, `select_star`, `wrong_jinja`

---

## Training

**Phase 1 — SFT (Unsloth LoRA)**

| Param | Value |
|-------|-------|
| Base model | Phi-3-mini-4k-instruct |
| LoRA r | 64 |
| LoRA alpha | 128 |
| Epochs | 3 |
| LR | 2e-4 |
| Batch (effective) | 16 |
| Scheduler | cosine |
| Quantization | None (full fp16) |

**Phase 2 — DPO (TRL)**

| Param | Value |
|-------|-------|
| Beta (KL penalty) | 0.1 |
| Epochs | 1 |
| LR | 5e-5 |

---

## Evaluation

| Metric | Base | SFT | DPO (target) |
|--------|------|-----|--------------|
| Exact Match | ~15% | ~55% | ~60% |
| Parse Accuracy | ~60% | ~90% | ~93% |

Evaluation uses 50 held-out examples in `eval/gold_set.jsonl` covering: QUALIFY, UNNEST, ARRAY_AGG, STRUCT, PERCENTILE_CONT, FORMAT_DATE, DATE_TRUNC, TIMESTAMP_DIFF, rolling windows, cohort retention, funnel analysis, and all dbt Jinja patterns.

---

## Inference

```bash
make serve   # FastAPI on :8000
make ui      # Streamlit playground
```

**POST `/generate`**
```json
{
  "schema": "table: orders\ncolumns:\n  - name: order_id  type: STRING",
  "question": "Return each customer's most recent order.",
  "dialect": "bigquery",
  "max_new_tokens": 512,
  "temperature": 0.1
}
```

---

## Stack

| Component | Library |
|-----------|---------|
| Base model | microsoft/Phi-3-mini-4k-instruct |
| SFT training | Unsloth + HuggingFace TRL |
| DPO training | TRL DPOTrainer |
| Data generation | Groq (llama-3.1-70b-versatile) |
| SQL validation | sqlglot (BigQuery dialect) |
| Inference API | FastAPI + uvicorn |
| Playground | Streamlit |
| Publishing | huggingface_hub |
| Deployment | Docker on Render (CPU) |

---

## Milestones

| # | Milestone | Output |
|---|-----------|--------|
| 1 | Schemas + templates | `data/schemas/`, `synthetic/*_templates.py` |
| 2 | Generate datasets | `data/dataset/sft_*.jsonl`, `data/dataset/dpo_pairs.jsonl` |
| 3 | SFT training | `models/sft-phi3-v1/` |
| 4 | DPO training | `models/dpo-phi3-v1/` |
| 5 | Benchmark | `eval/results/*.json` |
| 6 | Publish to Hub | `Mpratham/sql-forge-phi3-v1` |
| 7 | Deploy API + UI | Render service |
