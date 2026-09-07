"""Correctness tests for the inequity-aversion reward formula."""

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


def test_matches_hand_computed_formula_after_smoothing():
    r = InequityAversionReward(
        agent_ids=["a", "b"], alpha={"a": 2.0, "b": 2.0}, beta={"a": 0.2, "b": 0.2}, smoothing=0.9
    )
    r.apply({"a": 1.0, "b": 0.0})  # smoothed_a = 0.1, smoothed_b = 0.0
    out = r.apply({"a": 1.0, "b": 0.0})  # smoothed_a = 0.19, smoothed_b = 0.0
    smoothed_a = 0.9 * 0.1 + 0.1 * 1.0
    smoothed_b = 0.9 * 0.0 + 0.1 * 0.0
    assert r.smoothed_reward["a"] == pytest.approx(smoothed_a)
    assert r.smoothed_reward["b"] == pytest.approx(smoothed_b)
    # n=2, n-1=1: agent a is ahead -> pays beta * (smoothed_a - smoothed_b);
    # agent b is behind -> pays alpha * (smoothed_a - smoothed_b).
    diff = smoothed_a - smoothed_b
    assert out["a"] == pytest.approx(1.0 - 0.2 * diff)
    assert out["b"] == pytest.approx(0.0 - 2.0 * diff)


def test_agent_ahead_of_group_is_penalized_by_beta_not_alpha():
    r = InequityAversionReward(
        agent_ids=["a", "b", "c"],
        alpha={aid: 5.0 for aid in ("a", "b", "c")},
        beta={aid: 0.1 for aid in ("a", "b", "c")},
    )
    for _ in range(50):
        out = r.apply({"a": 10.0, "b": 0.0, "c": 0.0})
    # a is far ahead: small beta penalty keeps its adjusted reward close to
    # its raw reward, much closer than b/c's large alpha penalty pulls theirs.
    assert out["a"] > 5.0
    assert out["b"] < 0.0
    assert out["c"] < 0.0


def test_reset_clears_smoothed_state():
    r = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 1.0, "b": 1.0}, beta={"a": 1.0, "b": 1.0})
    r.apply({"a": 5.0, "b": 0.0})
    assert r.smoothed_reward["a"] != 0.0
    r.reset()
    assert r.smoothed_reward == {"a": 0.0, "b": 0.0}


def test_higher_alpha_pulls_disadvantaged_agent_reward_down_more():
    low = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 0.5, "b": 0.5}, beta={"a": 0.0, "b": 0.0})
    high = InequityAversionReward(agent_ids=["a", "b"], alpha={"a": 5.0, "b": 5.0}, beta={"a": 0.0, "b": 0.0})
    for _ in range(20):
        out_low = low.apply({"a": 0.0, "b": 1.0})
        out_high = high.apply({"a": 0.0, "b": 1.0})
    assert out_high["a"] < out_low["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
