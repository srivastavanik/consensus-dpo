#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Explain photosynthesis briefly.}"
MODEL="${2:-gpt-oss-small}"

for K in 3 5; do
  for M in 1 2; do
    for R in 0 1; do
      echo "Running K=$K M=$M R=$R"
      curl -s -X POST http://127.0.0.1:8000/consensus \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"model\":\"$MODEL\",\"k\":$K,\"m\":$M,\"r\":$R}" >/dev/null || true
    done
  done
done
echo "Ablations submitted. Check data/pairs.v1.jsonl"


