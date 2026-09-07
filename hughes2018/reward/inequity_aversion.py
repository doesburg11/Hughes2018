"""Inequity-averse intrinsic reward (Hughes et al. 2018, Sec. 3, Eq. 4).

A temporally-extended version of the Fehr & Schmidt (1999) inequity-aversion
utility function: each agent compares its reward not to others' *instantaneous*
per-step reward (which is dominated by noise from asynchronous individual
events -- one agent happening to pick up an apple this exact step, another
not), but to a discounted *trace* of everyone's past reward. The static
Fehr-Schmidt model compares final payoffs; this is the natural extension to
a setting where "payoff" is a reward stream, not a single number.

The paper's own Eq. 4 (verified against arXiv v3, arxiv.org/pdf/1803.08884 --
an earlier version of this file used a normalized exponential-moving-*average*
here instead, `e <- lambda*e + (1-lambda)*r`, which is a different, differently-
scaled quantity; caught in review against the primary text):

    e_i(t) = gamma * lambda * e_i(t-1) + r_i(t)                     (per step)

    r_i^total = r_i - (alpha_i / (n-1)) * sum_j max(e_j - e_i, 0)    # envy
                     - (beta_i  / (n-1)) * sum_j max(e_i - e_j, 0)   # guilt

This is an *unnormalized*, discount-accumulating trace (structurally an
eligibility-trace-like quantity, using the same `gamma` as the RL return
discount, plus a separate trace-decay `lambda`) -- not a bounded running
average. For a roughly constant reward rate, it settles near
`r / (1 - gamma*lambda)`, i.e. potentially several times larger in magnitude
than the raw reward itself, not the same scale. `alpha_i`/`beta_i` penalize
disadvantageous inequity (envy: others' trace is ahead) and advantageous
inequity (guilt: this agent's trace is ahead) respectively, sampled once per
agent (population heterogeneity), matching the same per-agent-parameter
design used for the reward term in the sibling SequentialSocialDilemmas
repo's `cleanup_reputation` experiment.

**Documented gap**: `alpha`/`beta`'s numeric ranges, and `lambda`'s exact
value, aren't reproduced here from this paper's primary text (see the
top-level README); `gamma` should match whatever discount factor the actor-
critic agent trains with, since the paper's trace reuses the RL discount.

**Known, deliberate simplification** (also documented in the README): the
paper additionally has each agent *observe* every other player's trace
`e_j` as part of its policy input, so the agent can react to inequity, not
just be reward-shaped by it. This repo's agents only receive the RGB
observation -- the inequity term still affects training via the reward
signal, but isn't observable to the policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def scaled_alpha_beta_range(
    gamma: float,
    trace_lambda: float,
    base_alpha_range: tuple[float, float] = (2.4, 3.0),
    base_beta_range: tuple[float, float] = (0.16, 0.20),
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Amplification-corrected alpha/beta sampling ranges.

    The base ranges above are the same `alpha ~ U(2.4, 3.0)`, `beta ~
    U(0.16, 0.20)` used for the reward term in the sibling
    SequentialSocialDilemmas repo's `cleanup_reputation` experiment -- tuned
    there for a bounded, reward-scale quantity. This trace is instead
    unnormalized and settles near `1 / (1 - gamma*trace_lambda)` times the
    raw reward's scale (module docstring), so applying those same ranges
    directly to *this* trace produces a penalty an order of magnitude too
    large relative to the extrinsic reward -- confirmed empirically (a short
    diagnostic run showed the inequity-averse condition's collective return
    staying far below, and not improving relative to, the baseline
    condition's, consistent with the penalty dominating the signal rather
    than shaping it). Dividing by the same amplification factor keeps the
    *effective* penalty on roughly the scale the base ranges were tuned for,
    regardless of which `gamma`/`trace_lambda` end up in use.

    This is a documented calibration choice, not a citation-backed number
    from Hughes et al. (2018)'s own primary text either way -- see the
    top-level README.
    """
    amplification = 1.0 / (1.0 - gamma * trace_lambda)
    alpha_lo, alpha_hi = base_alpha_range
    beta_lo, beta_hi = base_beta_range
    return (alpha_lo / amplification, alpha_hi / amplification), (beta_lo / amplification, beta_hi / amplification)


@dataclass
class InequityAversionReward:
    agent_ids: list[str]
    alpha: dict[str, float]
    beta: dict[str, float]
    gamma: float = 0.99  # should match the actor-critic's own discount factor
    trace_lambda: float = 0.95  # unstated by the paper as an exact number; documented, tunable default
    trace: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        missing_alpha = set(self.agent_ids) - set(self.alpha)
        missing_beta = set(self.agent_ids) - set(self.beta)
        if missing_alpha or missing_beta:
            raise ValueError(f"alpha/beta must be provided for every agent id; missing {missing_alpha | missing_beta}")
        self.trace = {agent_id: 0.0 for agent_id in self.agent_ids}

    def reset(self) -> None:
        self.trace = {agent_id: 0.0 for agent_id in self.agent_ids}

    def apply(self, raw_rewards: dict[str, float]) -> dict[str, float]:
        """Update each agent's reward trace and return inequity-adjusted rewards."""
        n = len(self.agent_ids)
        decay = self.gamma * self.trace_lambda
        for agent_id in self.agent_ids:
            r = raw_rewards.get(agent_id, 0.0)
            self.trace[agent_id] = decay * self.trace[agent_id] + r

        adjusted = dict(raw_rewards)
        if n <= 1:
            return adjusted
        for agent_id in self.agent_ids:
            own = self.trace[agent_id]
            disadvantageous = sum(max(self.trace[other] - own, 0.0) for other in self.agent_ids if other != agent_id)
            advantageous = sum(max(own - self.trace[other], 0.0) for other in self.agent_ids if other != agent_id)
            penalty = (self.alpha[agent_id] / (n - 1)) * disadvantageous + (
                self.beta[agent_id] / (n - 1)
            ) * advantageous
            adjusted[agent_id] = raw_rewards.get(agent_id, 0.0) - penalty
        return adjusted
