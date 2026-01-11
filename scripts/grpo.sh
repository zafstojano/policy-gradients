#!/bin/bash

uv run python -m pgrad.train \
    --loss_type grpo \
    --wandb_run_name "grpo_spell_backwards" \
    --beta 0.0
