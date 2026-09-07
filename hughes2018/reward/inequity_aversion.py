"""Inequity-averse intrinsic reward (Hughes et al. 2018, Sec. 3).

A temporally-extended version of the Fehr & Schmidt (1999) inequity-aversion
utility function: each agent compares its reward not to others' *instantaneous*
per-step reward (which is dominated by noise from asynchronous individual
events -- one agent happening to pick up an apple this exact step, another
not), but to a temporally-smoothed running estimate of everyone's reward. The
static Fehr-Schmidt model compares final payoffs; this is the natural
extension to a setting where "payoff" is a reward stream, not a single
number.

    smoothed_i <- lambda * smoothed_i + (1 - lambda) * r_i        (per step)

    r_i^total = r_i - (alpha_i / (n-1)) * sum_j max(smoothed_j - smoothed_i, 0)
                     - (beta_i  / (n-1)) * sum_j max(smoothed_i - smoothed_j, 0)

`alpha_i` penalizes disadvantageous inequity (envy: others are ahead);
`beta_i` penalizes advantageous inequity (guilt: this agent is ahead). Both
are sampled once per agent (population heterogeneity), matching the same
per-agent-parameter design used for the reputation reward in the sibling
SequentialSocialDilemmas repo's `cleanup_reputation` experiment.

**Documented gap**: the paper does not state an exact numeric smoothing
window for the temporal averaging (only that raw per-step comparison is too
noisy to use directly) -- `smoothing` below is a tunable parameter with a
default, not a verified reproduction of a specific unstated constant. This
mirrors how the sibling Leibo2017 repo handles its own unstated hyperparameters
(e.g. its epsilon-decay schedule length) -- made configurable and documented,
rather than presented as a settled number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InequityAversionReward:
    agent_ids: list[str]
    alpha: dict[str, float]
    beta: dict[str, float]
    smoothing: float = 0.95
    smoothed_reward: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        missing_alpha = set(self.agent_ids) - set(self.alpha)
        missing_beta = set(self.agent_ids) - set(self.beta)
        if missing_alpha or missing_beta:
            raise ValueError(f"alpha/beta must be provided for every agent id; missing {missing_alpha | missing_beta}")
        self.smoothed_reward = {agent_id: 0.0 for agent_id in self.agent_ids}

    def reset(self) -> None:
        self.smoothed_reward = {agent_id: 0.0 for agent_id in self.agent_ids}

    def apply(self, raw_rewards: dict[str, float]) -> dict[str, float]:
        """Update the smoothed-reward estimate and return inequity-adjusted rewards."""
        n = len(self.agent_ids)
        for agent_id in self.agent_ids:
            r = raw_rewards.get(agent_id, 0.0)
            self.smoothed_reward[agent_id] = (
                self.smoothing * self.smoothed_reward[agent_id] + (1.0 - self.smoothing) * r
            )

        adjusted = dict(raw_rewards)
        if n <= 1:
            return adjusted
        for agent_id in self.agent_ids:
            own = self.smoothed_reward[agent_id]
            disadvantageous = sum(
                max(self.smoothed_reward[other] - own, 0.0) for other in self.agent_ids if other != agent_id
            )
            advantageous = sum(
                max(own - self.smoothed_reward[other], 0.0) for other in self.agent_ids if other != agent_id
            )
            penalty = (self.alpha[agent_id] / (n - 1)) * disadvantageous + (
                self.beta[agent_id] / (n - 1)
            ) * advantageous
            adjusted[agent_id] = raw_rewards.get(agent_id, 0.0) - penalty
        return adjusted
