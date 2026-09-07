"""Independent advantage actor-critic agent, matching the network architecture
McKee et al. (2023) explicitly describe as inherited from this paper's own
setup ("convolutional neural network with 3x3 kernel, stride 1 and 32 output
channels, a two-layer multi-layer perceptron with 64 hidden units in each
layer, a long short-term memory (LSTM) of hidden size 128, and linear layers
for the policy logits and value function").

**Documented simplification**: Hughes et al. (2018) trains with A3C
(Asynchronous Advantage Actor-Critic, Mnih et al. 2016) -- multiple worker
processes each computing gradients independently and asynchronously against
a shared parameter server. This repo instead implements a synchronous
n-step advantage actor-critic: one rollout of fixed length is collected per
update, advantages/returns are computed with bootstrapping at the rollout's
end, and each agent's own network is updated independently (no parameter
sharing, no communication, matching the paper's independence assumption --
only the *async* part of A3C is simplified away, not the *independent actor-
critic per agent* part). This is the same kind of "own (much simpler) agent
architecture" simplification the sibling Leibo2017 repo makes for its own
independent-DQN setup vs. the paper's fuller method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Rollout:
    """One agent's fixed-length window of transitions, collected under
    `torch.no_grad()` via repeated `ActorCriticAgent.act()` calls, then
    replayed with gradients in `ActorCriticAgent.update()`."""

    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    initial_lstm_state: tuple = None
    bootstrap_value: float = 0.0


@dataclass
class ActorCriticConfig:
    num_actions: int
    conv_channels: int = 32
    fc_hidden: int = 64
    lstm_hidden: int = 128
    discount: float = 0.99  # matches this paper's own stated default (also used by McKee et al. 2023)
    learning_rate: float = 1e-4  # unspecified by the paper; own choice, see README
    entropy_coeff: float = 0.01  # unspecified by the paper; own choice, see README
    value_loss_coeff: float = 0.5
    grad_clip_norm: float = 5.0
    seed: int | None = None


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_channels: int, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv2d(obs_channels, cfg.conv_channels, kernel_size=3, stride=1, padding=1)
        self._flat_dim = None  # set lazily on first forward, since view size depends on obs H/W
        self.fc1 = nn.Linear(1, cfg.fc_hidden)  # in_features patched in _lazy_init
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.fc_hidden)
        self.lstm = nn.LSTMCell(cfg.fc_hidden, cfg.lstm_hidden)
        self.policy_head = nn.Linear(cfg.lstm_hidden, cfg.num_actions)
        self.value_head = nn.Linear(cfg.lstm_hidden, 1)

    def _lazy_init(self, flat_dim: int, device: torch.device) -> None:
        if self._flat_dim is not None:
            return
        self._flat_dim = flat_dim
        self.fc1 = nn.Linear(flat_dim, self.cfg.fc_hidden).to(device)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.cfg.lstm_hidden, device=device)
        c = torch.zeros(batch_size, self.cfg.lstm_hidden, device=device)
        return h, c

    def forward(self, obs: torch.Tensor, lstm_state: tuple[torch.Tensor, torch.Tensor]):
        """obs: (B, C, H, W) float32 in [0, 1]. Returns (policy_logits, value, new_lstm_state)."""
        x = F.relu(self.conv(obs))
        x = x.flatten(start_dim=1)
        self._lazy_init(x.shape[1], obs.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        h, c = self.lstm(x, lstm_state)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value, (h, c)


class ActorCriticAgent:
    """One independent learner: owns its network, optimizer, and LSTM state."""

    def __init__(self, obs_channels: int, config: ActorCriticConfig | None = None, device: str = "cpu"):
        self.cfg = config or ActorCriticConfig(num_actions=8)
        self.device = torch.device(device)
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        self.network = ActorCriticNetwork(obs_channels, self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)
        self.lstm_state = self.network.initial_state(1, self.device)

    def reset_lstm_state(self) -> None:
        self.lstm_state = self.network.initial_state(1, self.device)

    def act(self, obs) -> tuple[int, float, float, tuple[torch.Tensor, torch.Tensor]]:
        """Sample an action; returns (action, log_prob, value, lstm_state_before_this_step).

        The caller is expected to store `lstm_state_before_this_step` alongside
        the transition (needed to recompute the same forward pass with
        gradients during the update -- LSTM state can't be replayed from a
        detached snapshot taken *after* the step).
        """
        state_before = self.lstm_state
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits, value, new_state = self.network(x, state_before)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        self.lstm_state = (new_state[0].detach(), new_state[1].detach())
        return int(action.item()), float(log_prob.item()), float(value.item()), state_before

    def value_only(self, obs) -> float:
        """Estimate the value of the current LSTM state (for a rollout's
        bootstrap target) WITHOUT advancing `self.lstm_state` or sampling/
        consuming an action -- this observation is being used only to
        evaluate a target, not as a real step the agent took, so it must not
        perturb the persistent recurrent state the next rollout continues
        from."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        with torch.no_grad():
            _logits, value, _new_state = self.network(x, self.lstm_state)
        return float(value.item())

    def update(self, rollout: Rollout) -> dict[str, float]:
        """One synchronous n-step advantage actor-critic update from a full rollout.

        `rollout` holds this agent's own (obs, action, reward, done,
        lstm_state_before) for every step of the collection window, plus a
        bootstrap value for the state after the last step.
        """
        cfg = self.cfg
        device = self.device
        T = len(rollout.obs)

        returns = [0.0] * T
        running_return = rollout.bootstrap_value
        for t in reversed(range(T)):
            running_return = rollout.rewards[t] + cfg.discount * running_return * (1.0 - rollout.dones[t])
            returns[t] = running_return
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # Recompute the forward pass with gradients (the .act() calls that
        # produced this rollout ran under torch.no_grad()), carrying the LSTM
        # state forward across the window so gradients flow through the
        # recurrent connection (truncated BPTT over this rollout's length).
        state = (rollout.initial_lstm_state[0].to(device), rollout.initial_lstm_state[1].to(device))
        log_probs, values, entropies = [], [], []
        for t in range(T):
            x = torch.as_tensor(rollout.obs[t], dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            logits, value, state = self.network(x, state)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = torch.as_tensor([rollout.actions[t]], device=device)
            log_probs.append(dist.log_prob(action_t).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            values.append(value.squeeze(0))
            if rollout.dones[t]:
                state = self.network.initial_state(1, device)

        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        entropy_t = torch.stack(entropies).mean()

        advantages = (returns_t - values_t).detach()
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = F.mse_loss(values_t, returns_t)
        loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy_t

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_t.item()),
            "mean_return": float(returns_t.mean().item()),
        }
