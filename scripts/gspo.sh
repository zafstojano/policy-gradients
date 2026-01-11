#!/bin/bash

uv run python -m pgrad.train \
    --loss_type gspo \
    --compute_kl \
    --wandb_run_name "gspo_spell_backwards"
