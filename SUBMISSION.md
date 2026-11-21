# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

## Part A — Paper calculations

[Part A — paper calculations (PDF)](part_a.pdf)

## Part B — FrozenLake

### Fixed policy, written before the experiments

Actions: 0 = left, 1 = down, 2 = right, 3 = up. Holes: 5, 7, 11, 12. Goal: 15.

| State | Action | Reason |
| --- | --- | --- |
| 0 | down | start of my path 0→4→8→9→13→14→15 |
| 1 | left | back to 0; down is hole 5 |
| 2 | down | to 6, then down to the goal |
| 3 | left | to 2; down is hole 7 |
| 4 | down | path |
| 6 | down | to 10; left and right are holes |
| 8 | right | path; down is hole 12 |
| 9 | down | path |
| 10 | down | to 14, then to the goal |
| 13 | right | path |
| 14 | right | goal |

I designed this policy for the non-slippery map. It is the same in all experiments.

### Results

Gymnasium 1.4.0, NumPy 2.5.5

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0% | 7.94 |
| no | random | 22 | 1 | 49 | 0 | 2% | 7.16 |
| no | random | 33 | 2 | 48 | 0 | 4% | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| yes | random | 11 | 1 | 49 | 0 | 2% | 7.98 |
| yes | random | 22 | 3 | 47 | 0 | 6% | 7.66 |
| yes | random | 33 | 1 | 49 | 0 | 2% | 7.16 |
| yes | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| yes | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| yes | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Files:
- [episodes.csv](outputs/part_b/episodes.csv) — one row per episode (600 rows)
- [summary.md](outputs/part_b/summary.md) — this table
- [traces.txt](outputs/part_b/traces.txt) — example, successful and failed episodes (`*` = agent)

### What slipping changed

Without slipping, my fixed policy reached the goal in all 150 episodes, always in 6 steps.
With slipping, it reached the goal only 4 times out of 150.
On slippery ice, the agent moves in the chosen direction only 1/3 of the time.
In 1/3 + 1/3 of cases it slips to a side.
In states 1, 3, 4, 6, 8 and 10 one side is a hole, so a slip can end the episode.
My policy did not plan for this, so it was almost as bad as random actions.

### How consistent the batches were

The non-slippery fixed policy gave 100% in all three batches, because there is no randomness.
Other conditions were between 0% and 6%. The batches were close, but not equal,
because each batch uses different seeds. Time limits never happened: every episode ended in a goal or a hole before 100 steps.

### Why one successful episode is not enough

One success can be luck. For example, the slippery fixed policy reached the goal in episode 11020,
but it failed in 146 of 150 episodes. We need many episodes and several batches to see the real success rate.

### Example traces

- Successful: slippery, fixed policy, episode seed 11020, 14 steps.
- Failed: slippery, fixed policy, episode seed 11000, 8 steps. The agent chose "down" in state 4, slipped right and fell into hole 5.

### Why this is not training

The policy is fixed before the runs and never changes.
The agent does not use rewards to update its actions or learn anything.
We only measure how good the policy is. Training would change the policy based on experience.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

