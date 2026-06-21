"""
Advanced Soft Actor-Critic (SAC) Agent — Discrete Action Space.

Key upgrades over baseline:
  - N-step return targets (reduces variance)
  - Gradient clipping on all networks
  - Separate position-sizing output from actor
  - Dueling critic heads
  - Learning rate scheduling
  - Detailed per-update metrics logging
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, Dict, Optional
from collections import deque

from .networks import ActorNetwork, CriticNetwork
from ..memory.replay_buffer import PrioritizedReplayBuffer, NStepBuffer


class AdvancedSACAgent:
    """
    Advanced SAC agent with:
      - Twin dueling critics
      - Transformer-based actor
      - N-step return targets
      - Automatic entropy tuning
      - Position sizing output
      - Gradient norm clipping
      - Cosine LR annealing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sequence_length: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        action_values: np.ndarray,
        target_entropy_factor: float,
        initial_log_alpha: float,
        n_step: int = 3,
        grad_clip: float = 1.0,
        dropout: float = 0.1,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        use_position_sizing: bool = True,
        training_steps: int = 100_000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SAC] Using device: {self.device}")

        self.action_values = action_values
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.n_step = n_step
        self.grad_clip = grad_clip
        self.use_position_sizing = use_position_sizing

        # N-step discount
        self.gamma_n = gamma ** n_step

        # ── Networks ────────────────────────────────────────────
        net_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
        self.actor = ActorNetwork(**net_kwargs, use_position_sizing=use_position_sizing).to(self.device)

        self.critic_1 = CriticNetwork(**net_kwargs).to(self.device)
        self.critic_2 = CriticNetwork(**net_kwargs).to(self.device)
        self.target_critic_1 = CriticNetwork(**net_kwargs).to(self.device)
        self.target_critic_2 = CriticNetwork(**net_kwargs).to(self.device)
        self._copy_to_target(self.critic_1, self.target_critic_1)
        self._copy_to_target(self.critic_2, self.target_critic_2)

        # ── Optimisers ──────────────────────────────────────────
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=lr, weight_decay=1e-5,
        )

        # ── Temperature / Entropy ───────────────────────────────
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_factor
        self.log_alpha = torch.tensor(initial_log_alpha, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr)

        # ── LR Schedulers ───────────────────────────────────────
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=training_steps, eta_min=lr * 0.1)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=training_steps, eta_min=lr * 0.1)

        # ── Metrics tracking ────────────────────────────────────
        self.update_count = 0
        self.metrics_history: Dict[str, deque] = {
            k: deque(maxlen=100)
            for k in ["critic_loss", "actor_loss", "alpha_loss", "alpha", "entropy", "q_mean"]
        }

    # ── Public API ───────────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[float, int, float]:
        """
        Returns:
            action_value  — displacement value for TP/SL
            action_idx    — index into action_values
            position_size — fraction of capital [0, 1]
        """
        self.actor.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, pos_size = self.actor(s)

            if deterministic:
                action_idx = torch.argmax(logits, dim=1).item()
            else:
                dist = Categorical(logits=logits)
                action_idx = dist.sample().item()

        self.actor.train()
        return (
            float(self.action_values[action_idx]),
            int(action_idx),
            float(pos_size.item()),
        )

    def update(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        batch_size: int,
        beta: float,
    ) -> Dict[str, float]:
        """
        One gradient update step.
        Returns dict of metrics.
        """
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones, indices, weights = \
            replay_buffer.sample(batch_size, beta)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights     = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        alpha = self.log_alpha.exp()

        # ── Critic update ───────────────────────────────────────
        critic_loss, td_errors, q_mean = self._critic_loss(
            states, actions, rewards, next_states, dones, weights, alpha
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.grad_clip,
        )
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # Update PER priorities
        replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())

        # ── Actor update ────────────────────────────────────────
        actor_loss, entropy = self._actor_loss(states, alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # ── Alpha update ────────────────────────────────────────
        alpha_loss = self._alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ── Soft target update ──────────────────────────────────
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        self.update_count += 1

        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       alpha.item(),
            "entropy":     entropy.item(),
            "q_mean":      q_mean.item(),
        }
        for k, v in metrics.items():
            self.metrics_history[k].append(v)

        return metrics

    # ── Internal helpers ─────────────────────────────────────────────

    def _critic_loss(
        self,
        states, actions, rewards, next_states, dones, weights, alpha
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_logits, _ = self.actor(next_states)
            next_dist      = Categorical(logits=next_logits)
            next_probs     = next_dist.probs                          # [B, A]

            nq1 = self.target_critic_1(next_states)
            nq2 = self.target_critic_2(next_states)
            nq  = torch.min(nq1, nq2)                                # [B, A]

            # Soft value with entropy: V = Σ π(a|s') [Q(s',a) - α log π(a|s')]
            log_probs  = torch.log(next_probs + 1e-8)
            soft_v     = (next_probs * (nq - alpha * log_probs)).sum(dim=1, keepdim=True)

            # N-step target
            target_q   = rewards + (1.0 - dones) * self.gamma_n * soft_v

        # Current Q values
        q1 = self.critic_1(states).gather(1, actions.unsqueeze(1))  # [B, 1]
        q2 = self.critic_2(states).gather(1, actions.unsqueeze(1))

        td1 = q1 - target_q
        td2 = q2 - target_q
        td_errors = (torch.abs(td1) + torch.abs(td2)) / 2.0

        loss = (weights * (F.huber_loss(q1, target_q, reduction="none") +
                           F.huber_loss(q2, target_q, reduction="none"))).mean()

        return loss, td_errors, ((q1 + q2) / 2.0).mean()

    def _actor_loss(self, states, alpha) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.actor(states)
        dist      = Categorical(logits=logits)
        probs     = dist.probs
        log_probs = torch.log(probs + 1e-8)

        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        q  = torch.min(q1, q2)

        # Maximise expected Q - alpha * entropy
        actor_loss = (probs * (alpha * log_probs - q)).sum(dim=1).mean()
        entropy    = dist.entropy().mean()
        return actor_loss, entropy

    def _alpha_loss(self, states) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.actor(states)
            dist      = Categorical(logits=logits)
            log_probs = torch.log(dist.probs + 1e-8)

        # Encourage entropy ≥ target_entropy
        return -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

    @staticmethod
    def _copy_to_target(src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for tp, p in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(p.data)

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module) -> None:
        for tp, p in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "actor":           self.actor.state_dict(),
            "critic_1":        self.critic_1.state_dict(),
            "critic_2":        self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "log_alpha":       self.log_alpha.item(),
            "update_count":    self.update_count,
        }, path)
        print(f"[SAC] Saved checkpoint -> {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.log_alpha = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self.device
        )
        self.update_count = ckpt.get("update_count", 0)
        print(f"[SAC] Loaded checkpoint from {path}")

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def get_avg_metrics(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.metrics_history.items() if v}
