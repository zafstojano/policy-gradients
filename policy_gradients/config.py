from typing import Any

import yaml
from pydantic import BaseModel, model_validator


class DatasetSpec(BaseModel):
    name: str
    weight: int = 1
    config: dict[str, Any] = {}


class DataConfig(BaseModel):
    specs: list[DatasetSpec]
    size: int = 3000


class Config(BaseModel):
    data: DataConfig
    loss: str
    model_name: str = "Qwen/Qwen3-1.7B"
    clip_eps_lo: float = 0.2
    clip_eps_hi: float = 0.2
    clip_eps_val: float = 0.2
    beta: float = 0.0
    lr: float = 5e-6
    gamma: float = 0.99
    lam: float = 0.95
    vf_coef: float = 0.1
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_new_tokens: int = 512
    prompts_per_step: int = 4
    num_rollouts: int = 8
    rollout_batch_size: int = 8
    train_batch_size: int = 2
    batch_acc: int = 4
    max_norm: float = 1.0
    seed: int = 42
    model_device_id: int = 0
    ref_model_device_id: int = 1
    val_model_device_id: int = 2
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @model_validator(mode="after")
    def validate_rollout_batch_size(self) -> "Config":
        if self.num_rollouts > 1 and self.rollout_batch_size != self.num_rollouts:
            raise ValueError("When num_rollouts > 1, rollout_batch_size must equal num_rollouts.")
        return self


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
