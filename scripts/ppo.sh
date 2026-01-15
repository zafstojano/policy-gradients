#!/bin/bash

uv run python -m pgrad.train \
    --loss ppo \
    --wandb_run_name "ppo_spell_backwards" \
    --num_rollouts 1 \
    --prompts_per_step 40
