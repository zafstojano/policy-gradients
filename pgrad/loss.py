import torch
import torch.nn as nn

from .buffer import Experience


def approx_kl_div(log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """
    log_ratio = log_probs - log_probs_ref
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return (log_ratio.exp() - 1) - log_ratio


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)


class GRPOLoss(nn.Module):
    def __init__(self, clip_eps: float, beta: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.beta = beta

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> torch.Tensor:
        ratio = (log_probs - experience.log_probs_old).exp()
        unclipped_term = ratio * experience.advantages
        clipped_term = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * experience.advantages

        kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)

        loss = -torch.min(unclipped_term, clipped_term) + self.beta * kl_div
        loss = masked_mean(loss, experience.action_mask, dim=-1).mean(dim=0)
        kl_loss = masked_mean(kl_div.detach(), experience.action_mask, dim=-1).mean(dim=0)

        return loss, kl_loss


class GSPOLoss(nn.Module):
    def __init__(self, clip_eps: float, beta: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.beta = beta

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> torch.Tensor:
        seq_logprobs = masked_mean(
            log_probs - experience.log_probs_old, experience.action_mask, dim=-1, keepdim=True
        ).exp()
        unclipped_term = seq_logprobs * experience.advantages
        clipped_term = seq_logprobs.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * experience.advantages

        kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)

        loss = -torch.min(unclipped_term, clipped_term) + self.beta * kl_div
        loss = masked_mean(loss, experience.action_mask, dim=-1).mean(dim=0)
        kl_loss = masked_mean(kl_div.detach(), experience.action_mask, dim=-1).mean(dim=0)

        return loss, kl_loss
