FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU-only for inference serving on Render)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    transformers>=4.40.0 \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.6.0 \
    huggingface_hub>=0.22.0 \
    accelerate>=0.28.0

# Copy source
COPY inference/ ./inference/
COPY eval/ ./eval/

# Model is downloaded at startup from HuggingFace Hub
ENV MODEL_REPO=Mprtham/sql-forge-phi3-v1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
