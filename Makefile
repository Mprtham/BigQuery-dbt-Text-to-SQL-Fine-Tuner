.PHONY: help install generate validate train-sft train-dpo benchmark serve ui push clean

PYTHON := python

help:
	@echo "sql-forge — BigQuery+dbt SQL fine-tuning pipeline"
	@echo ""
	@echo "  make install          Install all Python dependencies"
	@echo "  make generate         Generate SFT + DPO datasets via Groq"
	@echo "  make generate-dry     Seed datasets from templates only (no API calls)"
	@echo "  make validate         Validate generated datasets before training"
	@echo "  make train-sft        Phase 1: SFT fine-tune with Unsloth"
	@echo "  make train-dpo        Phase 2: DPO alignment with TRL"
	@echo "  make benchmark        Run base vs SFT vs DPO evaluation"
	@echo "  make serve            Start FastAPI inference server"
	@echo "  make ui               Start Streamlit playground"
	@echo "  make push             Push model + datasets to HuggingFace Hub"
	@echo "  make clean            Remove generated datasets and model checkpoints"

install:
	pip install -r requirements.txt

generate-dry:
	$(PYTHON) synthetic/generate_sft.py --dry-run
	$(PYTHON) synthetic/generate_dpo.py --dry-run

generate:
	$(PYTHON) synthetic/generate_sft.py
	$(PYTHON) synthetic/generate_dpo.py

validate:
	$(PYTHON) scripts/validate_dataset.py

train-sft:
	$(PYTHON) train/sft_train.py

train-dpo:
	$(PYTHON) train/dpo_train.py

benchmark:
	$(PYTHON) eval/benchmark.py

serve:
	uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run inference/playground.py

push:
	$(PYTHON) scripts/push_to_hub.py --push-dataset --push-model

push-dataset:
	$(PYTHON) scripts/push_to_hub.py --push-dataset

push-model:
	$(PYTHON) scripts/push_to_hub.py --push-model

clean:
	rm -rf data/dataset/*.jsonl
	rm -rf models/sft-phi3-v1 models/dpo-phi3-v1
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
