#!/bin/bash

uv run python -m pgrad.train \
    --loss rloo \
    --wandb_run_name "rloo_spell_backwards"
