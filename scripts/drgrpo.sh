#!/bin/bash

uv run python -m pgrad.train \
    --loss drgrpo \
    --wandb_run_name "drgrpo_spell_backwards"
