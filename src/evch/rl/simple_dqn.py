from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from evch.models.common import make_mlp
from evch.utils.torch_runtime import resolve_torch_device


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        feature_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims
        feature_dim = hidden_dims[-1] if hidden_dims else 64
        self.feature_extractor = make_mlp(obs_dim, feature_dims, feature_dim)
        self.value_head = nn.Linear(feature_dim, 1)
        self.advantage_head = nn.Linear(feature_dim, action_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class SimpleDQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any], seed: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = resolve_torch_device(str(config.get("device", "auto")))
        self.rng = np.random.default_rng(seed)

        self.gamma = float(config["gamma"])
        self.batch_size = int(config["batch_size"])
        self.learning_starts = int(config["learning_starts"])
        self.target_update_interval = int(config["target_update_interval"])
        self.train_frequency = int(config["train_frequency"])
        self.gradient_steps = int(config.get("gradient_steps", 1))
        self.epsilon_start = float(config["epsilon_start"])
        self.epsilon_end = float(config["epsilon_end"])
        self.epsilon_decay_steps = int(config["epsilon_decay_steps"])
        self.tau = float(config.get("tau", 1.0))
        self.reward_clip = float(config.get("reward_clip", 0.0))
        self.heuristic_prior_strength = float(config.get("heuristic_prior_strength", 0.0))

        self.q_network = QNetwork(obs_dim, action_dim, list(config["hidden_dims"])).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, list(config["hidden_dims"])).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = Adam(self.q_network.parameters(), lr=float(config["learning_rate"]))
        self.replay_buffer = ReplayBuffer(int(config["replay_capacity"]))
        self.total_steps = 0

    def _valid_action_mask(self, env: Any | None = None) -> np.ndarray | None:
        if env is None or not hasattr(env, "valid_action_mask"):
            return None
        mask = np.asarray(env.valid_action_mask(), dtype=bool)
        if mask.shape != (self.action_dim,):
            raise ValueError("Environment valid_action_mask shape does not match action dimension.")
        return mask

    @staticmethod
    def _masked_argmax(q_values: np.ndarray, action_mask: np.ndarray | None) -> int:
        if action_mask is None:
            return int(np.argmax(q_values))
        valid_indices = np.flatnonzero(action_mask)
        if valid_indices.size == 0:
            return int(np.argmax(q_values))
        return int(valid_indices[int(np.argmax(q_values[valid_indices]))])

    def _heuristic_prior_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        if self.heuristic_prior_strength <= 0.0:
            return torch.zeros((observations.shape[0], self.action_dim), dtype=observations.dtype, device=observations.device)
        num_sites = self.action_dim - 1
        allocations = observations[:, :num_sites]
        normalized_site_scores = observations[:, 2 * num_sites : 3 * num_sites]
        occupied_scores = normalized_site_scores * allocations
        noop_prior = occupied_scores.max(dim=1, keepdim=True).values
        return torch.cat([normalized_site_scores, noop_prior], dim=1) * self.heuristic_prior_strength

    def _epsilon(self) -> float:
        progress = min(self.total_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: np.ndarray, deterministic: bool = False, env: Any | None = None) -> int:
        action_mask = self._valid_action_mask(env)
        if not deterministic and self.rng.random() < self._epsilon():
            if action_mask is None:
                return int(self.rng.integers(self.action_dim))
            valid_indices = np.flatnonzero(action_mask)
            if valid_indices.size > 0:
                return int(self.rng.choice(valid_indices))
            return int(self.rng.integers(self.action_dim))
        with torch.no_grad():
            tensor_obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = (self.q_network(tensor_obs) + self._heuristic_prior_tensor(tensor_obs)).squeeze(0).cpu().numpy()
        return self._masked_argmax(q_values, action_mask)

    def update(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        observations = torch.as_tensor(np.stack([t.observation for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.reward_clip > 0.0:
            rewards = torch.clamp(rewards, min=-self.reward_clip, max=self.reward_clip)
        next_observations = torch.as_tensor(
            np.stack([t.next_observation for t in batch]), dtype=torch.float32, device=self.device
        )
        next_action_masks = torch.as_tensor(
            np.stack([t.next_action_mask for t in batch]), dtype=torch.bool, device=self.device
        )
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(observations).gather(1, actions)
        with torch.no_grad():
            next_q_online = self.q_network(next_observations) + self._heuristic_prior_tensor(next_observations)
            masked_online = next_q_online.masked_fill(~next_action_masks, float("-inf"))
            next_actions = masked_online.argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_observations).gather(1, next_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self.tau < 1.0:
            self._update_target_network()
        return float(loss.item())

    def _update_target_network(self) -> None:
        if self.tau >= 1.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def train(self, env: Any, episodes: int, max_steps: int, run: Any | None = None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for episode in range(episodes):
            observation, _ = env.reset(seed=int(self.rng.integers(1_000_000)))
            episode_reward = 0.0
            served_total = 0.0
            unmet_total = 0.0
            invalid_actions = 0.0
            losses: list[float] = []
            for _ in range(max_steps):
                action = self.act(observation, deterministic=False, env=env)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_action_mask = self._valid_action_mask(env)
                if next_action_mask is None:
                    next_action_mask = np.ones(self.action_dim, dtype=bool)
                self.replay_buffer.push(
                    Transition(
                        observation=observation.copy(),
                        action=action,
                        reward=float(reward),
                        next_observation=next_observation.copy(),
                        next_action_mask=next_action_mask.copy(),
                        done=bool(terminated or truncated),
                    )
                )
                observation = next_observation
                episode_reward += float(reward)
                served_total += float(info["served_demand"])
                unmet_total += float(info["unmet_demand"])
                invalid_actions += float(not info.get("action_valid", True))
                self.total_steps += 1
                if self.total_steps % self.train_frequency == 0:
                    for _ in range(self.gradient_steps):
                        maybe_loss = self.update()
                        if maybe_loss is not None:
                            losses.append(maybe_loss)
                if self.total_steps % self.target_update_interval == 0 and self.tau >= 1.0:
                    self._update_target_network()
                if terminated or truncated:
                    break
            record = {
                "episode": float(episode),
                "reward": episode_reward,
                "served_demand": served_total,
                "unmet_demand": unmet_total,
                "invalid_actions": invalid_actions,
                "epsilon": self._epsilon(),
                "loss": float(np.mean(losses)) if losses else 0.0,
            }
            history.append(record)
            if run is not None:
                run.log({f"rl/{key}": value for key, value in record.items()})
        return history

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.q_network.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": self.config,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "SimpleDQNAgent":
        payload = torch.load(path, map_location=map_location)
        agent = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            config=dict(payload["config"]),
            seed=0,
        )
        agent.q_network.load_state_dict(payload["state_dict"])
        agent.target_network.load_state_dict(payload["state_dict"])
        return agent
