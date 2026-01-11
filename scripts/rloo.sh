#!/bin/bash

uv run python -m pgrad.train \
    --loss_type rloo \
    --compute_kl \
    --wandb_run_name "rloo_spell_backwards"
