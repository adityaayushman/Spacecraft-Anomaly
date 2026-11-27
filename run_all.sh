#!/bin/bash
# Worker: python -m sad.run_models for a list of "model:seed[:shard]" jobs (resumable)
for job in "$@"; do
  IFS=: read -r m s sh <<< "$job"
  python3 -m sad.run_models --model "$m" --seed "$s" --threads 2 --shard "${sh:-0/1}"
done
