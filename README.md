


# Policy Grads

A minimal hackable implementation of policy gradients for training large language models with reinforcement learning.



## Data
This project is using Reasoning Gym for generating procedural datasets. In the yaml file simply specify which datasets you want to use, along with their configurations. For example:

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

## Installation

```
# install uv
pip install uv

# sync the repo
uv sync

# Install flash attention separately
# look up the correct combination of flash attn, cuda, python and torch versions, and then download from https://github.com/mjun0812/flash-attention-prebuild-wheels/
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```
