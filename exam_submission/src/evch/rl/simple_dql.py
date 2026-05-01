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
from evch.rl.evaluation import evaluate_policy
from evch.utils.torch_runtime import resolve_torch_device


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.network = make_mlp(obs_dim, hidden_dims, action_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SimpleDQLAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any], seed: int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = resolve_torch_device(str(config.get("device", "auto")))
        self.rng = np.random.default_rng(seed)

        self.gamma = float(config["gamma"])
        self.learning_rate = float(config["learning_rate"])
        self.hidden_dims = [int(d) for d in config.get("hidden_dims", [128, 128])]
        self.epsilon_start = float(config["epsilon_start"])
        self.epsilon_end = float(config["epsilon_end"])
        self.epsilon_decay_steps = int(config["epsilon_decay_steps"])
        self.reward_clip = float(config.get("reward_clip", 0.0))
        self.max_grad_norm = float(config.get("max_grad_norm", 5.0))
        self.batch_size = max(int(config.get("batch_size", 64)), 1)
        self.replay_capacity = max(int(config.get("replay_capacity", 10000)), 1)
        self.learning_starts = max(int(config.get("learning_starts", 1)), 1)
        self.train_frequency = max(int(config.get("train_frequency", 1)), 1)
        self.gradient_steps = max(int(config.get("gradient_steps", 1)), 1)

        self.q_network = QNetwork(obs_dim, action_dim, self.hidden_dims).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer: deque[Transition] = deque(maxlen=self.replay_capacity)
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
        valid = np.flatnonzero(action_mask)
        if valid.size == 0:
            return int(np.argmax(q_values))
        return int(valid[int(np.argmax(q_values[valid]))])

    def _epsilon(self) -> float:
        progress = min(self.total_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: np.ndarray, deterministic: bool = False, env: Any | None = None) -> int:
        action_mask = self._valid_action_mask(env)
        if not deterministic and self.rng.random() < self._epsilon():
            if action_mask is None:
                return int(self.rng.integers(self.action_dim))
            valid = np.flatnonzero(action_mask)
            return int(self.rng.choice(valid)) if valid.size > 0 else int(self.rng.integers(self.action_dim))

        with torch.no_grad():
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(obs_t).squeeze(0).cpu().numpy()
        return self._masked_argmax(q_values, action_mask)

    def _td_loss_tensor(self, batch: list[Transition]) -> torch.Tensor:
        obs = torch.as_tensor(np.stack([t.observation for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.reward_clip > 0.0:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        next_obs = torch.as_tensor(np.stack([t.next_observation for t in batch]), dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.bool, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(obs).gather(1, actions)
        with torch.no_grad():
            next_q = self.q_network(next_obs).masked_fill(~next_masks, float("-inf"))
            target = rewards + self.gamma * (1.0 - dones) * next_q.max(dim=1, keepdim=True).values

        return torch.nn.functional.smooth_l1_loss(q_values, target)

    def _sample_replay_batch(self) -> list[Transition]:
        if not self.replay_buffer:
            return []
        buf = list(self.replay_buffer)
        if len(buf) <= self.batch_size:
            return buf
        indices = self.rng.choice(len(buf), size=self.batch_size, replace=False)
        return [buf[int(i)] for i in indices]

    def _update_from_batch(self, batch: list[Transition]) -> float:
        loss = self._td_loss_tensor(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def evaluate_td_loss(
        self,
        env: Any,
        episodes: int,
        seed: int,
        episode_seeds: list[int] | None = None,
    ) -> float:
        transitions: list[Transition] = []
        seeds = list(episode_seeds) if episode_seeds is not None else [seed + ep for ep in range(episodes)]
        for episode_seed in seeds:
            observation, _ = env.reset(seed=int(episode_seed))
            while True:
                action = self.act(observation, deterministic=True, env=env)
                next_observation, reward, terminated, truncated, _info = env.step(action)
                next_action_mask = self._valid_action_mask(env)
                if next_action_mask is None:
                    next_action_mask = np.ones(self.action_dim, dtype=bool)
                transitions.append(Transition(
                    observation=observation.copy(),
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation.copy(),
                    next_action_mask=next_action_mask.copy(),
                    done=bool(terminated or truncated),
                ))
                observation = next_observation
                if terminated or truncated:
                    break
        if not transitions:
            return 0.0
        with torch.no_grad():
            loss = self._td_loss_tensor(transitions)
        return float(loss.item())

    def train(
        self,
        env: Any,
        episodes: int,
        max_steps: int,
        eval_env_factory: Any | None = None,
        eval_interval: int = 0,
        eval_interval_steps: int = 0,
        eval_episodes: int = 1,
        eval_seed: int = 12345,
        eval_episode_seeds: list[int] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        next_eval_step = max(int(eval_interval_steps), 0) if eval_interval_steps > 0 else 0

        for episode in range(episodes):
            observation, _ = env.reset(seed=None)
            episode_reward = 0.0
            losses: list[float] = []

            for _ in range(max_steps):
                action = self.act(observation, deterministic=False, env=env)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_action_mask = self._valid_action_mask(env)
                if next_action_mask is None:
                    next_action_mask = np.ones(self.action_dim, dtype=bool)

                self.replay_buffer.append(Transition(
                    observation=observation.copy(),
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation.copy(),
                    next_action_mask=next_action_mask.copy(),
                    done=bool(terminated or truncated),
                ))
                observation = next_observation
                episode_reward += float(reward)
                self.total_steps += 1

                if self.total_steps >= self.learning_starts and self.total_steps % self.train_frequency == 0:
                    step_losses: list[float] = []
                    for _ in range(self.gradient_steps):
                        batch = self._sample_replay_batch()
                        if batch:
                            step_losses.append(self._update_from_batch(batch))
                    if step_losses:
                        losses.append(float(np.mean(step_losses)))

                if terminated or truncated:
                    break

            record: dict[str, float] = {
                "episode": float(episode),
                "reward": episode_reward,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": self._epsilon(),
                "eval_mean_reward": float("nan"),
                "eval_td_loss": float("nan"),
            }

            episode_interval_reached = eval_interval > 0 and ((episode + 1) % eval_interval == 0)
            step_interval_reached = eval_interval_steps > 0 and self.total_steps >= next_eval_step
            if eval_env_factory is not None and (step_interval_reached or episode_interval_reached or episode == episodes - 1):
                eval_env = eval_env_factory(eval_seed)
                eval_summary = evaluate_policy(
                    env=eval_env,
                    policy=lambda obs, inner_env, det=True: self.act(obs, deterministic=det, env=inner_env),
                    episodes=max(eval_episodes, 1),
                    seed=eval_seed,
                    deterministic=True,
                    episode_seeds=eval_episode_seeds,
                )
                if hasattr(eval_env, "close"):
                    eval_env.close()
                record["eval_mean_reward"] = float(eval_summary["mean_reward"])
                loss_env = eval_env_factory(eval_seed)
                eval_td_loss = self.evaluate_td_loss(
                    loss_env,
                    episodes=max(eval_episodes, 1),
                    seed=eval_seed,
                    episode_seeds=eval_episode_seeds,
                )
                if hasattr(loss_env, "close"):
                    loss_env.close()
                record["eval_td_loss"] = float(eval_td_loss)
                while eval_interval_steps > 0 and self.total_steps >= next_eval_step:
                    next_eval_step += eval_interval_steps

            history.append(record)
        return history

    def save(self, path: str | Path) -> None:
        torch.save({
            "state_dict": self.q_network.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": self.config,
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "SimpleDQLAgent":
        payload = torch.load(path, map_location=map_location)
        agent = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            config=dict(payload["config"]),
            seed=0,
        )
        agent.q_network.load_state_dict(payload["state_dict"])
        return agent


SimpleDQNAgent = SimpleDQLAgent
