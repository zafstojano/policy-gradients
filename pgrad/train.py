import argparse
import random
import re
import time

import numpy as np
import reasoning_gym as rg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import wandb

from .buffer import Experience, ReplayBuffer, join_experiences_batch
from .loss import CISPOLoss, GRPOLoss, GSPOLoss, RLOOLoss


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


def _accuracy_reward(dataset: ProceduralDataset, completions: str, entry: dict) -> float:
    def score_answer(completion: str) -> float:
        answer = extract_answer(completion)
        return dataset.score_answer(answer, entry)

    return [score_answer(c) for c in completions]


def _format_reward(completions: list[str], **kwargs) -> list[float]:
    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    dataset: ProceduralDataset, completions: list[str], entry: dict, format_weight: float = 0.5
) -> list[float]:
    accuracy_rewards = _accuracy_reward(dataset, completions, entry)
    format_rewards = _format_reward(completions)
    combined_rewards = [acc + format_weight * fmt for acc, fmt in zip(accuracy_rewards, format_rewards, strict=True)]
    return combined_rewards


@torch.no_grad()
def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


@torch.no_grad()
def compute_nonstandardized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - rewards.mean(dim=0, keepdim=True)


@torch.no_grad()
def compute_loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    K = rewards.shape[0]
    return (K / (K - 1)) * (rewards - rewards.mean(dim=0, keepdim=True))


@torch.no_grad()
def compute_returns(
    action_mask: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    B, S = action_mask.size()

    last_action_indices = action_mask.long().cumsum(dim=-1).argmax(dim=-1, keepdim=True)  # (B, 1)
    indices = torch.arange(S, device=action_mask.device).unsqueeze(0)  # (1, S)
    done = (indices >= last_action_indices).float()  # (B, S)

    rewards = torch.zeros_like(action_mask, device=action_mask.device, dtype=torch.float32).scatter_(
        dim=-1, index=last_action_indices, src=rewards
    )  # (B, S)
    returns = torch.zeros_like(action_mask, dtype=torch.float32, device=action_mask.device)  # (B, S)
    running = torch.zeros(B, device=action_mask.device, dtype=torch.float32)  # (B,)

    for t in reversed(range(S)):
        running = rewards[:, t] + gamma * (1.0 - done[:, t]) * running
        returns[:, t] = running

    returns = returns * action_mask
    return returns


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
    dataset: ProceduralDataset,
    tokenizer: AutoTokenizer,
    entry: dict,
    num_rollouts: int,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
):
    # 1. Format prompts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
        {"role": "user", "content": entry["question"]},
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
    rewards = compute_rewards(dataset, completions, entry)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=model.device).unsqueeze(-1)

    return sequence_ids, action_mask, rewards, completions


def main(args):
    # Init all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        model_device = f"cuda:{args.model_device_id}"
        ref_model_device = f"cuda:{args.ref_model_device_id}"
        val_model_device = f"cuda:{args.val_model_device_id}"
    else:
        model_device = ref_model_device = val_model_device = "cpu"

    dataset = rg.create_dataset(name=args.dataset_name, seed=args.seed, size=args.dataset_size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.prompts_per_step,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda x: x,
    )
    model, tokenizer = load_model(model_name=args.model_name, device_map=model_device)
    if args.compute_kl:
        ref_model, _ = load_model(model_name=args.model_name, device_map=ref_model_device)
        ref_model.eval()
    else:
        ref_model = None
    if args.loss_type in ["ppo"]:
        val_model, _ = load_model(model_name=args.model_name, device_map=val_model_device)
        val_model.lm_head = nn.Linear(val_model.lm_head.in_features, 1, bias=False, device=val_model.device)
        val_model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.loss_type in ["grpo", "drgrpo"]:
        objective = GRPOLoss(args.clip_eps_lo, args.clip_eps_hi, args.beta, args.compute_kl)
    elif args.loss_type == "gspo":
        objective = GSPOLoss(args.clip_eps_lo, args.clip_eps_hi, args.beta, args.compute_kl)
    elif args.loss_type == "rloo":
        objective = RLOOLoss(args.beta, args.compute_kl)
    elif args.loss_type == "cispo":
        objective = CISPOLoss(args.clip_eps_lo, args.clip_eps_hi, args.beta, args.compute_kl)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print("=" * 80)
    print(model)
    print("=" * 80)
    print()

    replay_buffer = ReplayBuffer()

    for step, batch in enumerate(dataloader):
        print(f"\n{'=' * 80}\n")
        print(f"[STEP {step}/{len(dataloader)}]")

        start = time.time()

        model.eval()
        replay_buffer.clear()
        rollout_rewards, rollout_completions = [], []

        for entry in batch:
            sequence_ids, action_mask, rewards, completions = rollout(
                model=model,
                dataset=dataset,
                tokenizer=tokenizer,
                entry=entry,
                num_rollouts=args.num_rollouts,
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )

            if args.loss_type in ["grpo", "gspo"]:
                advantages = compute_advantages(rewards)
            elif args.loss_type == "drgrpo":
                advantages = compute_nonstandardized_advantages(rewards)
            elif args.loss_type in ["rloo", "cispo"]:
                advantages = compute_loo_advantages(rewards)
            else:
                advantages = None

            returns = compute_returns(action_mask, rewards, gamma=args.gamma)

            attention_mask = sequence_ids != tokenizer.pad_token_id

            with torch.no_grad():
                log_probs_old = compute_log_probs(model, sequence_ids, attention_mask)
                log_probs_ref = compute_log_probs(ref_model, sequence_ids, attention_mask) if args.compute_kl else None

            experience = Experience(
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                action_mask=action_mask,
                returns=returns,
                advantages=advantages,
                log_probs_old=log_probs_old,
                log_probs_ref=log_probs_ref,
            ).to(cpu_device)
            replay_buffer.add(experience)

            rollout_rewards.append(rewards.detach().cpu())
            rollout_completions.append((entry["question"], entry["answer"], completions))

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

        end = time.time()
        print(f"\n  Step Time: {end - start:.2f} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="spell_backward")
    parser.add_argument("--dataset_size", type=int, default=3_000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--clip_eps_lo", type=float, default=0.2)
    parser.add_argument("--clip_eps_hi", type=float, default=0.2)
    parser.add_argument("--compute_kl", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--prompts_per_step", type=int, default=5)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--batch_acc", type=int, default=4)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="micro-pgrad")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--model_device_id", type=int, default=0)
    parser.add_argument("--ref_model_device_id", type=int, default=1)
    parser.add_argument("--val_model_device_id", type=int, default=2)
    parser.add_argument("--loss_type", type=str, choices=["grpo", "drgrpo", "gspo", "rloo", "cispo", "ppo"])
    args = parser.parse_args()

    main(args)
