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
        self.heuristic_prior_mode = str(config.get("heuristic_prior_mode", "placement")).lower()
        self.wandb_step_log_interval = int(config.get("wandb_step_log_interval", 1))

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
        if self.heuristic_prior_strength <= 0.0 or self.heuristic_prior_mode != "placement":
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

    def _td_loss_tensor(self, batch: list[Transition]) -> torch.Tensor:
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

        return torch.nn.functional.smooth_l1_loss(q_values, targets)

    def update(self) -> float | None:
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self._td_loss_tensor(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self.tau < 1.0:
            self._update_target_network()
        return float(loss.item())

    def evaluate_td_loss(self, env: Any, episodes: int, seed: int) -> float:
        transitions: list[Transition] = []
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            while True:
                action = self.act(observation, deterministic=True, env=env)
                next_observation, reward, terminated, truncated, _info = env.step(action)
                next_action_mask = self._valid_action_mask(env)
                if next_action_mask is None:
                    next_action_mask = np.ones(self.action_dim, dtype=bool)
                transitions.append(
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
                if terminated or truncated:
                    break

        if not transitions:
            return 0.0

        losses: list[float] = []
        for start in range(0, len(transitions), self.batch_size):
            batch = transitions[start : start + self.batch_size]
            losses.append(float(self._td_loss_tensor(batch).item()))
        return float(np.mean(losses)) if losses else 0.0

    def _update_target_network(self) -> None:
        if self.tau >= 1.0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def train(
        self,
        env: Any,
        episodes: int,
        max_steps: int,
        run: Any | None = None,
        eval_env_factory: Any | None = None,
        eval_interval: int = 0,
        eval_episodes: int = 1,
        eval_seed: int = 12345,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for episode in range(episodes):
            observation, _ = env.reset(seed=int(self.rng.integers(1_000_000)))
            episode_reward = 0.0
            served_total = 0.0
            unmet_total = 0.0
            invalid_actions = 0.0
            losses: list[float] = []
            action_trace: list[int] = []
            active_chargers_trace: list[float] = []
            mobile_stations_trace: list[float] = []
            utilization_trace: list[float] = []
            idle_capacity_trace: list[float] = []
            effective_capacity_trace: list[float] = []
            unused_mobile_chargers_trace: list[float] = []
            unused_mobile_stations_trace: list[float] = []
            activated_trace: list[float] = []
            adjusted_trace: list[float] = []
            queue_length_trace: list[float] = []
            queue_wait_trace: list[float] = []
            queue_wait_station_ab_trace: list[float] = []
            queue_wait_station_bc_trace: list[float] = []
            queue_wait_excess_trace: list[float] = []
            queue_wait_breach_trace: list[float] = []
            disruption_trace: list[float] = []
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
                action_trace.append(int(action))
                active_chargers_trace.append(float(info.get("num_active_chargers", 0.0)))
                mobile_stations_trace.append(float(info.get("num_active_mobile_stations", 0.0)))
                utilization_trace.append(float(info.get("utilization", 0.0)))
                idle_capacity_trace.append(float(info.get("idle_capacity", 0.0)))
                effective_capacity_trace.append(float(info.get("effective_capacity_total", 0.0)))
                unused_mobile_chargers_trace.append(float(info.get("unused_mobile_chargers", 0.0)))
                unused_mobile_stations_trace.append(float(info.get("unused_mobile_stations_estimate", 0.0)))
                activated_trace.append(float(info.get("activated_mobile_stations", 0.0)))
                adjusted_trace.append(float(info.get("adjusted_mobile_stations", 0.0)))
                queue_length_trace.append(float(info.get("queue_length", 0.0)))
                queue_wait_trace.append(float(info.get("queue_wait_mean_minutes", 0.0)))
                queue_wait_station_ab_trace.append(float(info.get("queue_wait_mean_minutes_station_ab", 0.0)))
                queue_wait_station_bc_trace.append(float(info.get("queue_wait_mean_minutes_station_bc", 0.0)))
                queue_wait_excess_trace.append(float(info.get("queue_wait_excess_minutes", 0.0)))
                queue_wait_breach_trace.append(float(info.get("queue_wait_target_breached", 0.0)))
                disruption_trace.append(float(info.get("disruption_active", 0.0)))
                self.total_steps += 1
                if run is not None and self.total_steps % max(self.wandb_step_log_interval, 1) == 0:
                    run.log(
                        {
                            "rl_step/action": float(action),
                            "rl_step/reward": float(reward),
                            "rl_step/served_demand": float(info["served_demand"]),
                            "rl_step/unmet_demand": float(info["unmet_demand"]),
                            "rl_step/active_mobile_stations": float(info.get("num_active_mobile_stations", 0.0)),
                            "rl_step/active_mobile_stations_station_ab": float(info.get("num_active_mobile_stations_station_ab", 0.0)),
                            "rl_step/active_mobile_stations_station_bc": float(info.get("num_active_mobile_stations_station_bc", 0.0)),
                            "rl_step/active_chargers": float(info.get("num_active_chargers", 0.0)),
                            "rl_step/unused_mobile_chargers": float(info.get("unused_mobile_chargers", 0.0)),
                            "rl_step/unused_mobile_stations_estimate": float(info.get("unused_mobile_stations_estimate", 0.0)),
                            "rl_step/queue_length_station_ab": float(info.get("queue_length_station_ab", 0.0)),
                            "rl_step/queue_length_station_bc": float(info.get("queue_length_station_bc", 0.0)),
                            "rl_step/queue_wait_mean_minutes_station_ab": float(info.get("queue_wait_mean_minutes_station_ab", 0.0)),
                            "rl_step/queue_wait_mean_minutes_station_bc": float(info.get("queue_wait_mean_minutes_station_bc", 0.0)),
                            "rl_step/unused_mobile_stations_estimate_station_ab": float(info.get("unused_mobile_stations_estimate_station_ab", 0.0)),
                            "rl_step/unused_mobile_stations_estimate_station_bc": float(info.get("unused_mobile_stations_estimate_station_bc", 0.0)),
                            "rl_step/utilization": float(info.get("utilization", 0.0)),
                            "rl_step/utilization_station_ab": float(info.get("utilization_station_ab", 0.0)),
                            "rl_step/utilization_station_bc": float(info.get("utilization_station_bc", 0.0)),
                            "rl_step/queue_length": float(info.get("queue_length", 0.0)),
                            "rl_step/queue_wait_mean_minutes": float(info.get("queue_wait_mean_minutes", 0.0)),
                            "rl_step/disruption_active": float(info.get("disruption_active", 0.0)),
                            "rl_step/disruption_type_code": float(info.get("disruption_type_code", 0.0)),
                            "rl_step/activated_mobile_stations": float(info.get("activated_mobile_stations", 0.0)),
                            "rl_step/adjusted_mobile_stations": float(info.get("adjusted_mobile_stations", 0.0)),
                        },
                        step=int(self.total_steps),
                    )
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
                "mean_action": float(np.mean(action_trace)) if action_trace else 0.0,
                "max_action": float(np.max(action_trace)) if action_trace else 0.0,
                "mean_active_chargers": float(np.mean(active_chargers_trace)) if active_chargers_trace else 0.0,
                "max_active_chargers": float(np.max(active_chargers_trace)) if active_chargers_trace else 0.0,
                "mean_active_mobile_stations": float(np.mean(mobile_stations_trace)) if mobile_stations_trace else 0.0,
                "max_active_mobile_stations": float(np.max(mobile_stations_trace)) if mobile_stations_trace else 0.0,
                "total_activated_mobile_stations": float(np.sum(activated_trace)) if activated_trace else 0.0,
                "total_adjusted_mobile_stations": float(np.sum(adjusted_trace)) if adjusted_trace else 0.0,
                "mean_utilization": float(np.mean(utilization_trace)) if utilization_trace else 0.0,
                "mean_idle_capacity": float(np.mean(idle_capacity_trace)) if idle_capacity_trace else 0.0,
                "mean_effective_capacity": float(np.mean(effective_capacity_trace)) if effective_capacity_trace else 0.0,
                "mean_unused_mobile_chargers": float(np.mean(unused_mobile_chargers_trace)) if unused_mobile_chargers_trace else 0.0,
                "mean_unused_mobile_stations_estimate": float(np.mean(unused_mobile_stations_trace)) if unused_mobile_stations_trace else 0.0,
                "mean_queue_length": float(np.mean(queue_length_trace)) if queue_length_trace else 0.0,
                "max_queue_length": float(np.max(queue_length_trace)) if queue_length_trace else 0.0,
                "mean_queue_wait_minutes": float(np.mean(queue_wait_trace)) if queue_wait_trace else 0.0,
                "mean_queue_wait_minutes_station_ab": float(np.mean(queue_wait_station_ab_trace)) if queue_wait_station_ab_trace else 0.0,
                "mean_queue_wait_minutes_station_bc": float(np.mean(queue_wait_station_bc_trace)) if queue_wait_station_bc_trace else 0.0,
                "mean_queue_wait_excess_minutes": float(np.mean(queue_wait_excess_trace)) if queue_wait_excess_trace else 0.0,
                "queue_wait_target_breach_fraction": float(np.mean(queue_wait_breach_trace)) if queue_wait_breach_trace else 0.0,
                "disruption_step_fraction": float(np.mean(disruption_trace)) if disruption_trace else 0.0,
            }
            should_eval = (
                eval_env_factory is not None
                and eval_interval > 0
                and (((episode + 1) % eval_interval == 0) or (episode == episodes - 1))
            )
            if should_eval:
                reward_env = eval_env_factory(eval_seed)
                eval_summary = evaluate_policy(
                    env=reward_env,
                    policy=lambda obs, inner_env, deterministic=True: self.act(obs, deterministic=deterministic, env=inner_env),
                    episodes=max(eval_episodes, 1),
                    seed=eval_seed,
                    deterministic=True,
                )
                if hasattr(reward_env, "close"):
                    reward_env.close()

                loss_env = eval_env_factory(eval_seed)
                eval_td_loss = self.evaluate_td_loss(loss_env, episodes=max(eval_episodes, 1), seed=eval_seed)
                if hasattr(loss_env, "close"):
                    loss_env.close()

                record.update(
                    {
                        "eval_mean_reward": float(eval_summary["mean_reward"]),
                        "eval_mean_served_demand": float(eval_summary["mean_served_demand"]),
                        "eval_mean_unmet_demand": float(eval_summary["mean_unmet_demand"]),
                        "eval_td_loss": float(eval_td_loss),
                    }
                )
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
