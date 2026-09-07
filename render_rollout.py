#!/usr/bin/env python3
# ruff: noqa: E402
"""Render a played-out Cleanup or Harvest episode to an animated GIF.

The graphics-utility counterpart to run_experiment*.py: those measure and
compare collective return, this lets you actually *look* at what a policy
does. Uses the env's own render() (a third-person, un-cropped RGB frame; see
GridWorldEnv.render() in grid_engine.py) upscaled with nearest-neighbor
resizing so individual cells stay crisp at GIF size.

Usage:
    python render_rollout.py --env cleanup                          # random policy (no --checkpoint-dir)
    python render_rollout.py --env cleanup --checkpoint-dir output/run_experiment1_cleanup/checkpoints/baseline
    python render_rollout.py --env harvest --checkpoint-dir output/run_experiment2_harvest/checkpoints/inequity-averse
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hughes2018.agents.actor_critic import ActorCriticAgent, ActorCriticConfig
from hughes2018.envs.cleanup import CleanupEnv
from hughes2018.envs.grid_engine import GridWorldConfig
from hughes2018.envs.harvest import HarvestEnv

ENV_CLASSES = {"cleanup": CleanupEnv, "harvest": HarvestEnv}


def build_agents(env, checkpoint_dir, device):
    """None if no checkpoint_dir is given -> caller falls back to random actions."""
    if checkpoint_dir is None:
        return None
    obs_hw = 2 * env.cfg.view_radius + 1
    agents = {}
    for agent_id in env.agents:
        agent = ActorCriticAgent(
            obs_channels=3,
            obs_height=obs_hw,
            obs_width=obs_hw,
            config=ActorCriticConfig(num_actions=env.num_actions),
            device=device,
        )
        checkpoint_path = Path(checkpoint_dir) / f"{agent_id}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
        agent.load(checkpoint_path)
        agents[agent_id] = agent
    return agents


def rollout(env, agents, num_steps, rng):
    obs = env.reset()  # fresh episode; env.__init__ already ran reset() once, this discards that state on purpose
    if agents is not None:
        for agent in agents.values():
            agent.reset_lstm_state()
    frames = [env.render()]
    for _ in range(num_steps):
        if agents is None:
            actions = {aid: int(rng.integers(env.num_actions)) for aid in env.agents}
        else:
            actions = {aid: agents[aid].act(obs[aid])[0] for aid in agents}
        obs, _rewards, dones, _infos = env.step(actions)
        frames.append(env.render())
        if dones.get("__all__", False):
            break
    return frames


def save_gif(frames, path, scale, fps):
    images = []
    for frame in frames:
        img = Image.fromarray(frame, mode="RGB")
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        images.append(img)
    duration_ms = int(1000 / fps)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_CLASSES), default="cleanup")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Omit for a random-policy rollout.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=16, help="Pixels per grid cell in the output GIF.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env_cls = ENV_CLASSES[args.env]
    env = env_cls(num_agents=args.num_agents, config=GridWorldConfig(episode_length=args.num_steps + 1), rng=rng)

    agents = build_agents(env, args.checkpoint_dir, args.device)
    frames = rollout(env, agents, args.num_steps, rng)

    out_path = args.out or f"output/render_rollout/{args.env}_{'random' if agents is None else 'trained'}.gif"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out_path, args.scale, args.fps)
    print(f"Wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
