import torch
import torch.nn as nn

from .buffer import Experience


def approx_kl_div(log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo approximation of KL divergence (k3 estimator). See: http://joschu.net/blog/kl-approx.html"""
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
    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, compute_kl: bool) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.compute_kl = compute_kl

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = (log_probs - experience.log_probs_old).exp()
        unclipped_term = ratio * experience.advantages
        clipped_term = ratio.clamp(1.0 - self.clip_eps_lo, 1.0 + self.clip_eps_hi) * experience.advantages
        loss = -torch.min(unclipped_term, clipped_term)

        if self.compute_kl:
            kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)
            loss = loss + self.beta * kl_div
            kl_loss = masked_mean(kl_div.detach(), mask=experience.action_mask, dim=-1).mean(dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device)

        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss, kl_loss


class GSPOLoss(nn.Module):
    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, compute_kl: bool) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.compute_kl = compute_kl

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        seq_logprobs = masked_mean(
            log_probs - experience.log_probs_old, mask=experience.action_mask, dim=-1, keepdim=True
        ).exp()
        unclipped_term = seq_logprobs * experience.advantages
        clipped_term = seq_logprobs.clamp(1.0 - self.clip_eps_lo, 1.0 + self.clip_eps_hi) * experience.advantages
        loss = -torch.min(unclipped_term, clipped_term)

        if self.compute_kl:
            kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)
            loss = loss + self.beta * kl_div
            kl_loss = masked_mean(kl_div.detach(), mask=experience.action_mask, dim=-1).mean(dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device)

        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss, kl_loss


class RLOOLoss(nn.Module):
    def __init__(self, beta: float, compute_kl: bool) -> None:
        super().__init__()
        self.beta = beta
        self.compute_kl = compute_kl

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        loss = -masked_mean(log_probs * experience.advantages, mask=experience.action_mask, dim=-1).mean(dim=0)

        if self.compute_kl:
            kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)
            loss = loss + self.beta * masked_mean(kl_div, experience.action_mask, dim=-1).mean(dim=0)
            kl_loss = masked_mean(kl_div.detach(), mask=experience.action_mask, dim=-1).mean(dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device)

        return loss, kl_loss


class CISPOLoss(nn.Module):
    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, beta: float, compute_kl: bool) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.compute_kl = compute_kl

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():  # stop gradient flow
            ratio = (log_probs - experience.log_probs_old).exp()
            clipped_ratio = ratio.clamp(1.0 - self.clip_eps_lo, 1.0 + self.clip_eps_hi)
        loss = -clipped_ratio * experience.advantages * log_probs

        if self.compute_kl:
            kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)
            loss = loss + self.beta * kl_div
            kl_loss = masked_mean(kl_div.detach(), mask=experience.action_mask, dim=-1).mean(dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=log_probs.device)

        loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        return loss, kl_loss


class PPOLoss(nn.Module):
    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, vf_coef: float, beta: float, compute_kl: bool) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.vf_coef = vf_coef
        self.beta = beta
        self.compute_kl = compute_kl

    def forward(
        self, log_probs: torch.Tensor, experience: Experience, values: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ratio = (log_probs - experience.log_probs_old).exp()
        # unclipped_term = ratio * experience.advantages
        # clipped_term = ratio.clamp(1.0 - self.clip_eps_lo, 1.0 + self.clip_eps_hi) * experience.advantages
        # loss = -torch.min(unclipped_term, clipped_term)

        # if self.compute_kl:
        #     kl_div = approx_kl_div(log_probs, experience.log_probs_ref, experience.action_mask)
        #     loss = loss + self.beta * kl_div
        #     kl_loss = masked_mean(kl_div.detach(), mask=experience.action_mask, dim=-1).mean(dim=0)
        # else:
        #     kl_loss = torch.tensor(0.0, device=log_probs.device)

        # loss = masked_mean(loss, mask=experience.action_mask, dim=-1).mean(dim=0)
        # return loss, kl_loss
        ...
