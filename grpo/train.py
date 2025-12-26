import argparse
import json
import random

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .buffer import ReplayBuffer


def get_dataloader(
    dataset_path: str,
    prompts_per_step: int,
    max_dataset_size: int | None = None,
    max_num_terms: int = 3,
    max_num_digits: int = 3,
    max_question_len: int = 128,
) -> DataLoader:
    predicate = (
        lambda x: x["num_terms"] <= max_num_terms
        and x["num_digits"] <= max_num_digits
        and len(x["question"]) < max_question_len
    )
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    dataset = [x for x in dataset if predicate(x)]
    if max_dataset_size is not None:
        dataset = dataset[:max_dataset_size]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=prompts_per_step,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    return dataloader


def load_model(
    model_name: str,
    trust_remote_code: bool = False,
    device_map=None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    return model, tokenizer


def main(args):
    # Init all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataloader = get_dataloader(
        dataset_path=args.dataset_path, prompts_per_step=args.prompts_per_step, max_dataset_size=args.prompts_per_step
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    model, tokenizer = load_model(model_name=args.model_name, device_map=device)
    ref_model, _ = load_model(model_name=args.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ref_model.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project)

    print(model)

    replay_buffer = ReplayBuffer()

    for step, batch in enumerate(dataloader):
        replay_buffer.clear()

        questions, answers = batch["question"], batch["answer"]
        print(questions)
        print(answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/math_tasks.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompts_per_step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--wandb_project", default=None)
    args = parser.parse_args()

    main(args)
