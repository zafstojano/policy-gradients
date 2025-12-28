import argparse
import json
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .buffer import Experience, ReplayBuffer, join_experiences_batch


SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively:
 <think> reasoning process here </think>
<answer> answer here </answer>
"""


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


def load_model(model_name: str, trust_remote_code: bool = False, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    return model, tokenizer


def compute_reward(completion: str, oracle_answer: str) -> float:
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL,
    )
    answer = answer_match.group(1) if answer_match else None
    reward = 0
    if answer is not None:
        answer = answer.strip()
        if answer == oracle_answer:
            reward = 1.0
        elif oracle_answer in answer:
            reward = 0.5
        else:
            reward = 0.01
    return reward


@torch.no_grad()
def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


@torch.no_grad()
def compute_log_probs(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return target_log_probs


@torch.no_grad()
def rollout(
    model,
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    num_rollouts: int,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    # 1. Format prompts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    messages_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer(
        messages_template,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(num_rollouts, 1)

    # 2. Generate responses
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        do_sample=True,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. Compute rewards
    rewards = [compute_reward(completion, answer) for completion in completions]
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

    return sequence_ids, action_mask, rewards, completions


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
    model_ref, _ = load_model(model_name=args.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    model_ref.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project)

    print(model)

    replay_buffer = ReplayBuffer()

    for step, batch in enumerate(dataloader):
        print(f"Step {step}")
        replay_buffer.clear()

        questions, answers = batch["question"], batch["answer"]

        for q, a in zip(questions, answers, strict=True):
            sequence_ids, action_mask, rewards, completions = rollout(
                model=model,
                tokenizer=tokenizer,
                question=q,
                answer=a,
                num_rollouts=args.num_rollouts,
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )

            advantages = compute_advantages(rewards)
            attention_mask = sequence_ids != tokenizer.pad_token_id

            log_probs_old = compute_log_probs(model, sequence_ids, attention_mask)
            log_probs_ref = compute_log_probs(model_ref, sequence_ids, attention_mask)

            experience = Experience(
                sequence_ids=sequence_ids,
                log_probs_old=log_probs_old,
                log_probs_ref=log_probs_ref,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
            ).to(cpu_device)
            replay_buffer.add(experience)

        torch.cuda.empty_cache()
        experience_sampler = DataLoader(
            dataset=replay_buffer.buffer,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=join_experiences_batch,
        )

        for epoch in range(args.epochs_per_step):
            for experience in experience_sampler:
                print(f"Experience: {experience}")
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/math_tasks.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompts_per_step", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=8)
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
