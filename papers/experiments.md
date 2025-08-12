Consensus → DPO: Experiments Plan (Skeleton)

- Data creation: K∈{3,5,7}, R∈{0,1,2}, m∈{1,2,3}; judge bias controls on/off
- Filters: drop ties/inconsistent, verbosity penalty, citation requirement for fact tasks
- Training: β∈{0.1,0.2,0.5}, lr=1e-5, wd=0.01, max_len 8–16k, global batch≥128 via grad accum
- Evaluation: GSM8K, MMLU, (optional) GPQA; hallucination via citation verification
- Telemetry: MLflow for params/metrics/artifacts

Fill this with results from `/data/runs/` and the UI dashboards.

