"""Correctness test for run_experiment1_cleanup.py's inequity_mode wiring.

Specifically guards the "advantageous_only" mode, which is what makes this
script actually test the paper's own claim about Cleanup (Fig. 3(A-C):
advantageous inequity aversion / guilt alone helps; Fig. 3(D-F):
disadvantageous / envy alone does not) -- a silently-wrong alpha here (e.g.
sampling it instead of zeroing it) would make this experiment test a
condition the paper never validated, without erroring or even necessarily
looking wrong in a training curve.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment1_cleanup import build_inequity_reward  # noqa: E402


def test_advantageous_only_zeroes_alpha_but_keeps_beta_positive():
    agent_ids = [f"agent-{i}" for i in range(5)]
    reward = build_inequity_reward(agent_ids, "advantageous_only", trace_lambda=0.95, seed=0)
    assert reward is not None
    assert all(reward.alpha[aid] == 0.0 for aid in agent_ids)
    assert all(reward.beta[aid] > 0.0 for aid in agent_ids)


def test_both_mode_samples_positive_alpha_and_beta():
    agent_ids = [f"agent-{i}" for i in range(5)]
    reward = build_inequity_reward(agent_ids, "both", trace_lambda=0.95, seed=0)
    assert reward is not None
    assert all(reward.alpha[aid] > 0.0 for aid in agent_ids)
    assert all(reward.beta[aid] > 0.0 for aid in agent_ids)


def test_none_mode_returns_no_reward_object():
    agent_ids = [f"agent-{i}" for i in range(5)]
    assert build_inequity_reward(agent_ids, "none", trace_lambda=0.95, seed=0) is None


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        build_inequity_reward(["agent-0"], "bogus_mode", trace_lambda=0.95, seed=0)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
