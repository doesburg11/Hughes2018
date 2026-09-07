"""Correctness tests for the inequity-aversion reward formula (Eq. 4:
e_i(t) = gamma * lambda * e_i(t-1) + r_i(t), an unnormalized discounted
trace -- NOT a bounded running average, so these tests avoid hardcoding
absolute-magnitude expectations that assumed the earlier (incorrect)
EWMA-average version of this formula; see the module docstring."""

import pytest

from hughes2018.reward.inequity_aversion import InequityAversionReward


def test_requires_alpha_beta_for_every_agent():
    with pytest.raises(ValueError):
        InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 1.0}, beta={"a": 1.0, "b": 1.0})


def test_single_agent_is_a_no_op():
    r = InequityAversionReward(agent_ids=["a"], alpha={"a": 5.0}, beta={"a": 5.0})
    out = r.apply({"a": 3.0})
    assert out == {"a": 3.0}


def test_equal_rewards_produce_no_penalty():
    r = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 2.0, "b": 2.0}, beta={"a": 0.2, "b": 0.2})
    for _ in range(5):
        out = r.apply({"a": 1.0, "b": 1.0})
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(1.0)


def test_matches_hand_computed_trace_formula():
    # gamma=1.0, trace_lambda=0.9 for clean arithmetic (decay = 0.9).
    r = InequityAversionReward(
        agent_ids=["a", "b"], alpha={"a": 2.0, "b": 2.0}, beta={"a": 0.2, "b": 0.2}, gamma=1.0, trace_lambda=0.9
    )
    r.apply({"a": 1.0, "b": 0.0})  # e_a = 0.9*0 + 1.0 = 1.0, e_b = 0.0
    out = r.apply({"a": 1.0, "b": 0.0})  # e_a = 0.9*1.0 + 1.0 = 1.9, e_b = 0.0
    assert r.trace["a"] == pytest.approx(1.9)
    assert r.trace["b"] == pytest.approx(0.0)
    # n=2, n-1=1: agent a's trace is ahead -> pays its own beta * diff;
    # agent b's trace is behind -> pays its own alpha * diff.
    diff = 1.9 - 0.0
    assert out["a"] == pytest.approx(1.0 - 0.2 * diff)
    assert out["b"] == pytest.approx(0.0 - 2.0 * diff)


def test_agent_ahead_of_group_pays_smaller_relative_penalty_with_small_beta():
    # Compare each agent's *penalty magnitude* (raw reward minus adjusted
    # reward), not an absolute adjusted-reward threshold: the trace's
    # steady-state scale depends on gamma/trace_lambda, so only the relative
    # comparison (small beta vs. large alpha) is a robust thing to assert.
    r = InequityAversionReward(
        agent_ids=["a", "b", "c"],
        alpha={aid: 5.0 for aid in ("a", "b", "c")},
        beta={aid: 0.1 for aid in ("a", "b", "c")},
    )
    for _ in range(50):
        out = r.apply({"a": 10.0, "b": 0.0, "c": 0.0})
    penalty_a = 10.0 - out["a"]  # a is ahead -> pays its small beta
    penalty_b = 0.0 - out["b"]  # b is behind -> pays its large alpha
    penalty_c = 0.0 - out["c"]
    assert penalty_a < penalty_b
    assert penalty_a < penalty_c
    assert penalty_b > 0.0 and penalty_c > 0.0


def test_reset_clears_trace_state():
    r = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 1.0, "b": 1.0}, beta={"a": 1.0, "b": 1.0})
    r.apply({"a": 5.0, "b": 0.0})
    assert r.trace["a"] != 0.0
    r.reset()
    assert r.trace == {"a": 0.0, "b": 0.0}


def test_higher_alpha_pulls_disadvantaged_agent_reward_down_more():
    low = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 0.5, "b": 0.5}, beta={"a": 0.0, "b": 0.0})
    high = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 5.0, "b": 5.0}, beta={"a": 0.0, "b": 0.0})
    for _ in range(20):
        out_low = low.apply({"a": 0.0, "b": 1.0})
        out_high = high.apply({"a": 0.0, "b": 1.0})
    assert out_high["a"] < out_low["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
