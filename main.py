import torch


if __name__ == "__main__":
    gamma = 0.99
    B, L = 2, 7
    rewards = torch.zeros(B, L)
    done_mask = torch.zeros(B, L)

    rewards[0, -1] = 1.0
    rewards[1, -2] = 0.5
    done_mask[0, -1:] = 1.0
    done_mask[1, -2:] = 1.0

    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(L)):
        running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
        returns[:, t] = running

    print("Rewards:\n", rewards)
    print("Done Mask:\n", done_mask)
    print("Returns:\n", returns)
