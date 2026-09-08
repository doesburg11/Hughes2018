"""Synchronous multi-agent training loop: N independent actor-critic learners
sharing one environment, with an optional inequity-aversion reward applied to
each step's raw rewards before they're stored for each agent's own update.

See `hughes2018/agents/actor_critic.py`'s module docstring for how this
loop's synchronous rollout collection relates to (and simplifies) the
paper's asynchronous A3C.

Also computes and threads through the per-step observable inequity-trace
vector (Sec. 3.2: "we allow agents to observe the smoothed reward of every
player on each timestep") for any agent configured with
`ActorCriticConfig.trace_dim > 0` -- this happens regardless of whether
`inequity_reward` is set, so an agent's observation shape stays identical
across conditions (a zero-filled vector when there's no inequity reward).
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
    agent_id_order = list(agents.keys())  # fixed order for the shared trace vector, every step
    for agent in agents.values():
        agent.reset_lstm_state()
    if inequity_reward is not None:
        inequity_reward.reset()

    episode_collective_return = 0.0

    def current_observable_trace() -> list[float]:
        # Sec. 3.2: "we allow agents to observe the smoothed reward of every
        # player on each timestep." Every agent sees the same vector (in
        # agent_id_order) regardless of its own trace_dim -- act()/
        # value_only() only actually use it for agents configured with
        # trace_dim > 0 (see ActorCriticConfig.trace_dim). Zero-filled when
        # there's no inequity_reward at all, so the vector's shape (and
        # therefore any trace_dim > 0 agent's observation shape) stays
        # identical across conditions, baseline included.
        if inequity_reward is not None:
            return inequity_reward.observable_trace()
        return [0.0] * len(agent_id_order)

    while stats.total_steps < num_env_steps:
        rollouts = {
            agent_id: Rollout(initial_lstm_state=agents[agent_id].lstm_state) for agent_id in agents
        }

        for _ in range(rollout_length):
            obs_before_step = obs  # o_t: what act() below actually conditions on
            # Captured *before* this step's env.step()/inequity_reward.apply()
            # runs -- see InequityAversionReward.observable_trace()'s
            # docstring for why this ordering is what keeps it causally e^{t-1}.
            observable_trace = current_observable_trace()
            actions = {}
            for agent_id, agent in agents.items():
                action, _log_prob, _value, _state_before = agent.act(obs_before_step[agent_id], observable_trace)
                actions[agent_id] = action

            obs, raw_rewards, dones, _infos = env.step(actions)  # obs is now o_{t+1}
            rewards = inequity_reward.apply(raw_rewards) if inequity_reward is not None else raw_rewards

            episode_collective_return += sum(raw_rewards.values())
            stats.total_steps += 1

            done = dones.get("__all__", False)
            for agent_id in agents:
                # Store o_t (not the post-step o_{t+1}) paired with a_t: this
                # is what update() replays through the network to recompute
                # log_prob(a_t | o_t) with gradients. Pairing a_t with
                # o_{t+1} instead (a real bug caught in review, since fixed)
                # would silently train on the wrong state-action pairing and
                # shift the whole LSTM replay sequence by one step.
                rollouts[agent_id].obs.append(obs_before_step[agent_id])
                rollouts[agent_id].actions.append(actions[agent_id])
                rollouts[agent_id].rewards.append(rewards.get(agent_id, 0.0))
                rollouts[agent_id].dones.append(bool(done))
                rollouts[agent_id].traces.append(observable_trace)

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
                # The trace passed here is *this* moment's (post-window)
                # observable trace -- the same e^t that would be fed as
                # e^{t-1} into the observation for the actual next action.
                rollout.bootstrap_value = agent.value_only(obs[agent_id], current_observable_trace())
            log = agent.update(rollout)
            log["agent_id"] = agent_id
            log["total_steps"] = stats.total_steps
            stats.update_logs.append(log)

        if on_log is not None and len(stats.update_logs) % (log_every_updates * len(agents)) < len(agents):
            on_log(stats)

    return stats
