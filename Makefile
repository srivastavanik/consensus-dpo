PY=python

.PHONY: dev ui orch retriever workers dpo ablate dvc-init ci

dev:
	pip install -e .
	pip install -r requirements-dev.txt || true

ui:
	ORCH_URL=http://127.0.0.1:8000 streamlit run apps/ui-dashboard/app.py

orch:
	uvicorn apps.orchestrator.main:app --reload

retriever:
	uvicorn apps.retriever.main:app --reload --port 8010

workers:
	redis-server --daemonize yes || true
	$(PY) apps/generator-worker/main.py & $(PY) apps/judge-worker/main.py

dpo:
	STUDENT_MODEL=gpt2 DPO_PAIRS_PATH=./data/pairs.v1.jsonl $(PY) apps/trainer/train_dpo.py

ablate:
	bash training/scripts/ablate.sh "Explain photosynthesis briefly." gpt-oss-small

dvc-init:
	pip install dvc
	dvc init -f || true
	dvc add data/pairs.v1.jsonl || true

ci:
	ruff check .


