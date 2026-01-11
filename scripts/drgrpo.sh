#!/bin/bash

uv run python -m pgrad.train \
    --loss_type drgrpo \
    --compute_kl \
    --wandb_run_name "drgrpo_spell_backwards"
