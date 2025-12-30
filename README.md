# Installation

```
# install uv
pip install uv

# sync the repo
uv sync

# Install flash attention separately
# look up the correct combination of flash attn, cuda, python and torch versions, and then download from https://github.com/mjun0812/flash-attention-prebuild-wheels/
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```
