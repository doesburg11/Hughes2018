# Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas — Hughes et al. (2018)

> **This is a from-scratch paper reproduction, not a ported codebase.** This repo rebuilds Hughes et al. (2018)'s Cleanup and Harvest environments and its inequity-aversion reward mechanism directly from the paper, with its own independent actor-critic training and no dependency on Vinitsky et al.'s code. For an engineering port of Vinitsky's existing `sequential_social_dilemma_games` codebase to Ray RLlib's new API stack instead — which already includes a simpler, instantaneous-reward version of this same mechanism as an opt-in flag (`inequity_averse_reward`) — see the sibling repo **[SequentialSocialDilemmas](https://github.com/doesburg11/SequentialSocialDilemmas)**. For the from-scratch reproduction of the *earlier* DeepMind SSD paper (Leibo et al. 2017, Gathering and Wolfpack, no Cleanup/Harvest at all), see **[Leibo2017](https://github.com/doesburg11/Leibo2017)**. Three repos with overlapping lineage and easy-to-confuse names — this note, and matching notes in the other two, are here so none of them gets mistaken for another.

A from-scratch replication of:

> Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzmán, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K. R., Koster, R., Roff, H., & Graepel, T. (2018). *Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas.* NeurIPS 2018. https://arxiv.org/abs/1803.08884

The paper's question: purely self-interested reinforcement learners tend to fail sequential social dilemmas like Cleanup and Harvest (the individually rational move — never clean, always harvest — collapses the shared resource for everyone). Fehr & Schmidt (1999)'s inequity-aversion model explains a lot of human cooperative behavior in one-shot economic games with a simple idea: people don't just want reward, they're also averse to *inequity* — doing worse than others (envy) and, to a lesser extent, doing better than others (guilt). This paper asks whether giving that same aversion to independent reinforcement-learning agents, extended from a single-payoff comparison to a *reward stream* comparison, produces the same cooperation-enabling effect in temporally extended, spatial social dilemmas — and finds that it does. Cleanup and Harvest, both introduced in this paper, later became the standard testbeds this whole line of DeepMind work builds on (Vinitsky et al.'s port, and McKee et al. 2023's reputation experiment reused in the sibling SequentialSocialDilemmas repo, both use these same two environments).

## The mechanism

Each agent's actual training reward is `r_i^total = r_i - inequity_penalty_i`:

```
smoothed_i <- lambda * smoothed_i + (1 - lambda) * r_i                    (every step)

inequity_penalty_i = (alpha_i / (n-1)) * sum_j max(smoothed_j - smoothed_i, 0)   # envy
                    + (beta_i  / (n-1)) * sum_j max(smoothed_i - smoothed_j, 0)  # guilt
```

`alpha_i` (envy weight) and `beta_i` (guilt weight) are sampled once per agent at construction — population heterogeneity, not resampled per episode, matching the same design used for the reward term in the sibling SequentialSocialDilemmas repo's `cleanup_reputation` experiment (built earlier this project, and itself built on `inequity_averse_reward`'s existing shape in that repo's `map_env.py`).

**The one thing this repo gets right that the simpler port doesn't**: SequentialSocialDilemmas' `inequity_averse_reward` compares agents' *raw, instantaneous per-step* rewards. The paper is explicit that this is too noisy to be meaningful — Fehr-Schmidt's original model compares *final payoffs*, and the natural temporally-extended analogue is a *smoothed* reward stream, not a single noisy step (one agent happening to step onto an apple this exact tick, another not, says nothing about who's actually "ahead"). `smoothed_i` above is that smoothing (`InequityAversionReward` in `hughes2018/reward/inequity_aversion.py`) — see `test_matches_hand_computed_formula_after_smoothing` in `tests/test_inequity_aversion.py` for the exact arithmetic, verified against a hand-computed example.

## What's matched vs. simplified

**Matched, from the paper's own stated design:**
- Both environments' core dynamics: Cleanup's piecewise-linear apple-regrowth-vs-river-pollution curve (`THRESHOLD_DEPLETION=0.4`, `THRESHOLD_RESTORATION=0.0`, `WASTE_SPAWN_PROBABILITY=0.5`, `APPLE_RESPAWN_PROBABILITY=0.05`) and Harvest's local-density-dependent regrowth (`SPAWN_PROBABILITY_BY_NEIGHBOR_COUNT = (0.0, 0.005, 0.02, 0.05)` indexed by nearby-apple count in a Moore neighborhood) — these are the paper's own environment mechanics (independently corroborated by Vinitsky et al.'s port and McKee et al. 2023's reuse of the same Cleanup design, not copied from either).
- The shared action set (move forward/backward/strafe-left/strafe-right relative to current facing, turn left/right, stay, a beam) and 3-cell-wide, 5-cell-long beam geometry.
- The network architecture explicitly stated (by McKee et al. 2023's Materials and Methods) to be inherited from this paper's own setup: 3x3-kernel/32-channel conv, two 64-unit FC layers, a 128-unit LSTM, linear policy/value heads.
- Per-agent independent actor-critic learners, no parameter sharing, no communication — the paper's stated independence assumption.
- The inequity-aversion formula's shape (temporally-smoothed envy/guilt terms, normalized by `n-1`).

**Not published by the paper, and chosen here (documented, not hidden — the same "blind spots" convention the sibling Leibo2017 repo uses for its own unstated details):**
- **Exact map layouts for Cleanup and Harvest.** Neither paper publishes pixel-exact level files. `hughes2018/envs/cleanup.py`/`harvest.py` use their own map designs with the right qualitative structure (a spatially separate river and orchard for Cleanup; an open apple field for Harvest), not a copy of any existing implementation's specific ASCII map.
- **Synchronous n-step actor-critic instead of asynchronous A3C.** The paper trains with true A3C (Mnih et al. 2016): multiple worker processes computing gradients independently and asynchronously against a shared parameter server. This repo collects one fixed-length rollout at a time and updates each agent's own network synchronously — same independent-per-agent actor-critic method, simplified execution model. See `hughes2018/agents/actor_critic.py`'s module docstring.
- **The inequity-aversion temporal smoothing window (`lambda`).** The paper states that raw per-step comparison is too noisy and a smoothed comparison is used instead, but doesn't give an exact numeric constant for it. `InequityAversionReward.smoothing` (default `0.95`) is a documented, tunable parameter, not a verified reproduction of an unstated number.
- **Learning rate, entropy coefficient, and `alpha`/`beta` sampling ranges** are not quoted from the paper's primary text here; the `alpha ~ U(2.4, 3.0)`, `beta ~ U(0.16, 0.20)` default used in the run scripts is carried over from the same range used for a related reward term in the sibling SequentialSocialDilemmas repo's `cleanup_reputation` experiment (itself following the general shape this lineage of papers typically uses), not a citation-backed number from this specific paper.
- **Movement-conflict resolution.** Two agents can't occupy the same cell; conflicts are resolved by processing agents in a random per-step order (an agent moves into its target cell only if unoccupied at that moment, otherwise stays put). This is simpler than Vinitsky's multi-pass algorithm (which additionally lets two agents swap places in one step) and isn't specified by the paper either way.

## Running it

```bash
git clone git@github.com:doesburg11/Hughes2018.git
cd Hughes2018
./create_conda_env.sh
conda activate ./.conda
# or: pip install -r requirements.txt into any Python 3.11 environment

pytest tests/ -q
```

Every `run_experiment*.py` script defaults to a small step count that exercises the full pipeline (env → independent actor-critic training → inequity-aversion reward → comparison) as a smoke test — it does not claim converged, paper-scale results. Pass `--total-steps` with a much larger value (and patience, or a GPU via `--device cuda`) to attempt that.

```bash
python run_experiment1_cleanup.py     # Cleanup: baseline vs. inequity-averse
python run_experiment2_harvest.py     # Harvest: baseline vs. inequity-averse
python run_experiment3_heterogeneous.py   # mixed population: some inequity-averse, some selfish
```

Each writes `results.json` (and, if `matplotlib` is available, a comparison plot) to `output/<script-name>/`.

## Known gaps from the paper

- No attempt at the paper's actual training scale (millions of steps per agent across an async multi-worker A3C setup) — see "Running it" above.
- `run_experiment3_heterogeneous.py` is a first-order version of the paper's population-heterogeneity question (do inequity-averse agents get exploited by selfish co-players in a single fixed-composition training run), not its full evolutionary/generational-selection treatment across which trait survives over many generations — that's a documented follow-up, not built here.
- No convolutional-architecture ablation or hyperparameter sweep reproducing the paper's own robustness checks (if it has any beyond the headline Cleanup/Harvest comparison — not verified against the primary text in this pass).

## Tests

`tests/` covers, in order of dependency: the shared gridworld primitives (movement, rotation, beam geometry, observation orientation — the same "sign verified against beam/move directions" rigor the sibling Leibo2017 repo uses for its own primitives), each environment's paper-specific mechanics (Cleanup's pollution/regrowth curve, Harvest's density-dependent spawning), the inequity-aversion reward formula (including a hand-computed exact-value check), the actor-critic agent (including a hand-computed discounted-return check), and an end-to-end training smoke test. Run with `pytest tests/ -q`.

## Acknowledgments

Developed with AI coding assistance from [Claude](https://claude.com/claude-code) (Anthropic), which does the implementation, with [Codex](https://openai.com/codex) (OpenAI) acting as an independent second opinion, peer-reviewing Claude's nontrivial code changes.

## References

- Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzmán, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K. R., Koster, R., Roff, H., & Graepel, T. (2018). [Inequity Aversion Improves Cooperation in Intertemporal Social Dilemmas](https://arxiv.org/abs/1803.08884). NeurIPS 2018.
- Fehr, E., & Schmidt, K. M. (1999). [A Theory of Fairness, Competition, and Cooperation](https://doi.org/10.1162/003355399556151). The Quarterly Journal of Economics, 114(3), 817-868.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). ICML 2016. (The A3C algorithm this paper trains with, and this repo's own actor-critic simplifies.)
- McKee, K. R., Hughes, E., Zhu, T. O., Chadwick, M. J., Koster, R., Castañeda, A. G., Beattie, C., Graepel, T., Botvinick, M., & Leibo, J. Z. (2023). [A Multi-Agent Reinforcement Learning Model of Reputation and Cooperation in Human Groups](https://arxiv.org/abs/2103.04982). arXiv:2103.04982. (Reuses this paper's Cleanup design; source of the network-architecture spec cited above.)
- Vinitsky, E., Jaques, N., Leibo, J., Castañeda, A., Hughes, E., et al. [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games) — the open-source port this project's sibling repo, SequentialSocialDilemmas, updates to Ray's new API stack.
