#!/bin/bash

uv run python -m pgrad.train \
    --loss_type cispo \
    --wandb_run_name "cispo_spell_backwards" \
    --clip_eps_lo 0.2 \
    --clip_eps_hi 0.4
