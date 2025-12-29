import argparse
import json
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import wandb

from .buffer import Experience, ReplayBuffer, join_experiences_batch
from .loss import GRPOLoss


SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively:
<think>
reasoning process here
</think>
<answer>
answer here
</answer>
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


def compute_log_probs(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
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

    # 3. Obtain the generated tokens only
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 4. Compute rewards
    rewards = [compute_reward(completion, answer) for completion in completions]
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

    return sequence_ids, action_mask, rewards, completions


def main(args):
    # Init all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        model_device, ref_model_device = f"cuda:{args.model_device_id}", f"cuda:{args.ref_model_device_id}"
    else:
        model_device, ref_model_device = "cpu", "cpu"

    dataloader = get_dataloader(dataset_path=args.dataset_path, prompts_per_step=args.prompts_per_step)
    model, tokenizer = load_model(model_name=args.model_name, device_map=model_device)
    model_ref, _ = load_model(model_name=args.model_name, device_map=ref_model_device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    objective = GRPOLoss(args.clip_eps, args.beta)

    model_ref.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, config=vars(args))

    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)
    print("=" * 80)
    print()

    replay_buffer = ReplayBuffer()

    for step, batch in enumerate(dataloader):
        print(f"\n{'=' * 80}")
        print(f"STEP {step}")
        print("=" * 80)

        model.eval()
        replay_buffer.clear()
        rollout_rewards, rollout_completions = [], []

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

            with torch.no_grad():
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

            rollout_rewards.append(rewards.cpu())
            rollout_completions.append((q, a, completions))

        avg_reward = torch.cat(rollout_rewards, dim=0).mean().item()
        wandb.log({"avg_reward": avg_reward})
        print("\n  Rollout Results:")
        print(f"    Average Reward: {avg_reward:.4f}")

        # Print a sample completion
        if rollout_completions:
            sample_q, sample_a, sample_completions = rollout_completions[0]
            sample_completion = sample_completions[0]

            # Truncate if needed
            max_len = 1000
            if len(sample_completion) > max_len:
                sample_completion = sample_completion[:max_len] + "..."

            print("\n  Sample Completion:")
            print(f"    Question: {sample_q}")
            print(f"    Oracle Answer: {sample_a}")
            print(f"    Model Completion:\n{sample_completion}")

        torch.cuda.empty_cache()
        model.train()
        experience_sampler = DataLoader(
            dataset=replay_buffer.buffer,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=join_experiences_batch,
        )

        print(f"\n  Training ({args.epochs_per_step} epoch(s)):")
        for epoch in range(args.epochs_per_step):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_kl_loss = 0.0

            for batch_idx, experience in enumerate(experience_sampler):
                experience: Experience
                experience = experience.to(model.device)

                log_probs = compute_log_probs(model, experience.sequence_ids, experience.attention_mask)
                loss, kl_loss = objective(log_probs, experience)

                if not loss.isfinite():
                    print(
                        f"    [Epoch {epoch + 1}/{args.epochs_per_step}, Batch {batch_idx + 1}] WARNING: Loss is inf!"
                    )
                    continue

                # Scale loss by accumulation steps
                scaled_loss = loss / args.batch_acc
                scaled_loss.backward()

                accumulated_loss += loss.item()
                accumulated_kl_loss += kl_loss.item()

                # Update weights every batch_acc steps
                if (batch_idx + 1) % args.batch_acc == 0 or (batch_idx + 1) == len(experience_sampler):
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    # Log averaged metrics
                    num_accumulated = min(args.batch_acc, (batch_idx % args.batch_acc) + 1)
                    avg_loss = accumulated_loss / num_accumulated
                    avg_kl_loss = accumulated_kl_loss / num_accumulated

                    wandb.log(
                        {
                            "loss": avg_loss,
                            "kl_loss": avg_kl_loss,
                            "grad_norm": grad_norm,
                        }
                    )
                    print(
                        f"    [Epoch {epoch + 1}/{args.epochs_per_step}, Batch {batch_idx + 1}] "
                        f" Loss: {avg_loss:.4f} | KL: {avg_kl_loss:.4f} | Grad Norm: {grad_norm:.4f}"
                    )

                    # Reset accumulators
                    accumulated_loss = 0.0
                    accumulated_kl_loss = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/math_tasks.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--prompts_per_step", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--batch_acc", type=int, default=3)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="micro-grpo")
    parser.add_argument("--model_device_id", type=int, default=0)
    parser.add_argument("--ref_model_device_id", type=int, default=3)
    args = parser.parse_args()

    main(args)
