#!/bin/bash

uv run python -m pgrad.train \
    --loss reinforce \
    --wandb_run_name "reinforce_spell_backwards" \
    --num_rollouts 1 \
    --prompts_per_step 40
