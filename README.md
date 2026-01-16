


# Policy Grads

A minimal hackable implementation of policy gradients for training large language models with RL.



## Data
This project is using [Reasoning Gym](https://github.com/open-thought/reasoning-gym) for generating procedural datasets. In the yaml file simply specify which datasets you want to use, along with their configurations. For example:

```yaml
data:
  size: 3000
  specs:
    - name: spell_backward
      weight: 1
      config:
        min_word_len: 3
        max_word_len: 10
    # To add more datasets in the mixture, simply list them here
    # - name: leg_counting
    #   weight: 1
```

See the full gallery of available datasets [here](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md).

## Installation

For dependencies, this project uses [uv](https://docs.astral.sh/uv/). Once installed, you can sync the repository and install dependencies with:

```
uv sync
```

In order to install flash attention, please go to this [repo](https://github.com/mjun0812/flash-attention-prebuild-wheels/) in order to find a wheel that matches your CUDA, Python and Torch versions. For example, installing Flash Attention 2.8.3 for CUDA 12.8, Python 3.12 and Torch 2.9 can be done with:

```
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

>[!WARNING]
> If you re-run `uv sync`, you may need to re-install flash attention with the above command.

## Development

To run the main training script, say with GRPO:
```
uv run python -m policy_grads.train --config configs/grpo.yaml
```

To run the linter
```
uv run pre-commit run --all-files
```
