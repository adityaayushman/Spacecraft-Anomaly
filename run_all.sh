#!/bin/bash
# Worker script: python -m sad.run_models for a list of "model:seed" jobs (resumable)
for job in "$@"; do python3 -m sad.run_models --model ${job%:*} --seed ${job#*:} --threads 2; done
