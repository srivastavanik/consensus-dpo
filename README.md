## consensus-dpo

Research-grade system for multi-agent consensus (generation → debate → judge) distilled via DPO into a deployable student model. All inference uses Novita AI GPT-OSS endpoints; training uses DPO/QLoRA.

### Quick start
- Copy `.env.example` to `.env` and set your Novita credentials
- Install core deps: `pip install -e .[service]`
- Run orchestrator: `uvicorn apps.orchestrator.main:app --reload`

### Layout
See `consensus-dpo/` for apps and libs. Services are decoupled and can run locally or via Docker/K8s.

### Status
Bootstrap commit: provider abstractions, Novita client, orchestrator and worker stubs, DPO trainer scaffold, retrieval stub.


