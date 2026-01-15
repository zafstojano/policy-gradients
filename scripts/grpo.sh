#!/bin/bash

uv run python -m pgrad.train \
    --loss grpo \
    --wandb_run_name "grpo_spell_backwards"
