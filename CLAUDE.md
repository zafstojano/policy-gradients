# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


To run the linter
```
uv run pre-commit run --all-files
```

To run the main training script, say with grpo:
```
uv run train --config configs/grpo.yaml
```
Never run this yourself directly to verify the implementation
