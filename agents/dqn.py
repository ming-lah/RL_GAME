from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.replay_buffer import ReplayBuffer


def _build_mlp(input_dim: int, hidden_layers: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden_layers: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_interval: int = 500
    warmup_steps: int = 2_000
    gradient_clip: float = 5.0


class DQNAgent:
    def __init__(self, config: DQNConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = _build_mlp(config.state_dim, config.hidden_layers, config.action_dim).to(self.device)
        self.target_net = _build_mlp(config.state_dim, config.hidden_layers, config.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        self.training_steps = 0

    def act(self, state, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.config.action_dim))

        if not isinstance(state, torch.Tensor):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if len(self.memory) < self.config.warmup_steps:
            return None

        batch = self.memory.sample(self.config.batch_size)
        states = torch.from_numpy(batch["states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def set_eval_mode(self) -> None:
        self.policy_net.eval()
        self.target_net.eval()

    def set_train_mode(self) -> None:
        self.policy_net.train()
        self.target_net.eval()

    def save(self, path: Path) -> None:
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> "DQNAgent":
        payload = torch.load(path, map_location=device or torch.device("cpu"))
        config = DQNConfig(**payload["config"])
        agent = cls(config, device=device)
        agent.policy_net.load_state_dict(payload["policy_state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.epsilon = payload.get("epsilon", config.epsilon_end)
        agent.set_eval_mode()
        return agent
