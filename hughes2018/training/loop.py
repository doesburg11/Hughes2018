"""Synchronous multi-agent training loop: N independent actor-critic learners
sharing one environment, with an optional inequity-aversion reward applied to
each step's raw rewards before they're stored for each agent's own update.

See `hughes2018/agents/actor_critic.py`'s module docstring for how this
loop's synchronous rollout collection relates to (and simplifies) the
paper's asynchronous A3C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hughes2018.agents.actor_critic import ActorCriticAgent, Rollout
from hughes2018.reward.inequity_aversion import InequityAversionReward


@dataclass
class TrainingStats:
    total_steps: int = 0
    total_episodes: int = 0
    episode_returns: list = field(default_factory=list)  # one collective-return float per completed episode
    update_logs: list = field(default_factory=list)  # one dict per rollout update, per agent


def train(
    env,
    agents: dict[str, ActorCriticAgent],
    num_env_steps: int,
    rollout_length: int = 20,
    inequity_reward: InequityAversionReward | None = None,
    log_every_updates: int = 50,
    on_log=None,
) -> TrainingStats:
    stats = TrainingStats()
    obs = env.reset()
    for agent in agents.values():
        agent.reset_lstm_state()
    if inequity_reward is not None:
        inequity_reward.reset()

    episode_collective_return = 0.0

    while stats.total_steps < num_env_steps:
        rollouts = {
            agent_id: Rollout(initial_lstm_state=agents[agent_id].lstm_state) for agent_id in agents
        }

        for _ in range(rollout_length):
            actions = {}
            action_book_keeping = {}
            for agent_id, agent in agents.items():
                action, _log_prob, _value, _state_before = agent.act(obs[agent_id])
                actions[agent_id] = action
                action_book_keeping[agent_id] = action

            obs, raw_rewards, dones, _infos = env.step(actions)
            rewards = inequity_reward.apply(raw_rewards) if inequity_reward is not None else raw_rewards

            episode_collective_return += sum(raw_rewards.values())
            stats.total_steps += 1

            done = dones.get("__all__", False)
            for agent_id in agents:
                rollouts[agent_id].obs.append(_none_to_prev(rollouts[agent_id].obs, obs, agent_id))
                rollouts[agent_id].actions.append(action_book_keeping[agent_id])
                rollouts[agent_id].rewards.append(rewards.get(agent_id, 0.0))
                rollouts[agent_id].dones.append(bool(done))

            if done:
                stats.total_episodes += 1
                stats.episode_returns.append(episode_collective_return)
                episode_collective_return = 0.0
                obs = env.reset()
                for agent in agents.values():
                    agent.reset_lstm_state()
                if inequity_reward is not None:
                    inequity_reward.reset()
                break  # end this rollout window early at an episode boundary

            if stats.total_steps >= num_env_steps:
                break

        for agent_id, agent in agents.items():
            rollout = rollouts[agent_id]
            if not rollout.obs:
                continue
            if rollout.dones[-1]:
                rollout.bootstrap_value = 0.0
            else:
                # value_only() reads the agent's current (already-advanced-
                # by-this-rollout) LSTM state without mutating it or
                # consuming an action, so the next rollout's initial state
                # still matches exactly what the agent actually observed.
                rollout.bootstrap_value = agent.value_only(obs[agent_id])
            log = agent.update(rollout)
            log["agent_id"] = agent_id
            log["total_steps"] = stats.total_steps
            stats.update_logs.append(log)

        if on_log is not None and len(stats.update_logs) % (log_every_updates * len(agents)) < len(agents):
            on_log(stats)

    return stats


def _none_to_prev(existing_obs_list, obs_dict, agent_id):
    """`obs` is keyed by agent id and updated every step; this just indexes
    into it defensively in case an agent is momentarily absent from the dict
    (not currently possible with these envs, since every agent gets an
    observation every step including while removed/tagged-out, but guards
    against a future env where that isn't true)."""
    if agent_id in obs_dict:
        return obs_dict[agent_id]
    return existing_obs_list[-1] if existing_obs_list else None
