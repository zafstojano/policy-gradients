#!/bin/bash

uv run python -m pgrad.train \
    --loss gspo \
    --wandb_run_name "gspo_spell_backwards"
