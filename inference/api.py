"""
FastAPI inference endpoint for sql-forge.
POST /generate → BigQuery or dbt SQL.
"""

import time
from pathlib import Path
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "models" / "dpo-phi3-v1"
FALLBACK   = ROOT / "models" / "sft-phi3-v1"

app = FastAPI(title="sql-forge", version="1.0")

# ── Lazy model loading ────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else str(FALLBACK)
    if not Path(model_path).exists():
        raise RuntimeError(f"No model found at {MODEL_DIR} or {FALLBACK}. Run training first.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map  = "auto",
    )
    _pipeline = pipeline(
        "text-generation",
        model          = model,
        tokenizer      = tokenizer,
        pad_token_id   = tokenizer.eos_token_id,
    )
    return _pipeline


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    schema: str
    question: str
    dialect: Literal["bigquery", "dbt"] = "bigquery"
    max_new_tokens: int = 512
    temperature: float = 0.1


class GenerateResponse(BaseModel):
    sql: str
    tokens_used: int
    latency_ms: int
    model: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.schema.strip():
        raise HTTPException(status_code=400, detail="schema is required")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    prompt = (
        f"### Schema:\n{req.schema.strip()}\n\n"
        f"### Question:\n{req.question.strip()}\n\n"
        f"### SQL:"
    )

    pipe = get_pipeline()
    t0 = time.time()
    outputs = pipe(
        prompt,
        max_new_tokens = req.max_new_tokens,
        do_sample      = req.temperature > 0.0,
        temperature    = req.temperature if req.temperature > 0.0 else 1.0,
        return_full_text = False,
    )
    latency_ms = int((time.time() - t0) * 1000)

    raw_sql = outputs[0]["generated_text"].strip()
    # Trim after first blank line to avoid trailing hallucinations
    sql = raw_sql.split("\n\n")[0].strip()

    tokens_used = len(pipe.tokenizer.encode(prompt + sql))
    model_name  = "sql-forge-dpo-phi3-v1" if MODEL_DIR.exists() else "sql-forge-sft-phi3-v1"

    return GenerateResponse(
        sql         = sql,
        tokens_used = tokens_used,
        latency_ms  = latency_ms,
        model       = model_name,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
