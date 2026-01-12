#!/bin/bash

uv run python -m pgrad.train \
    --loss_type drgrpo \
    --wandb_run_name "drgrpo_spell_backwards"
