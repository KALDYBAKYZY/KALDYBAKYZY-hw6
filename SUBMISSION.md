# HW6 submission

## AI-use declaration

I used Claude (Anthropic) for:
- explaining the MDP lecture and the value iteration formula in simple words;
- checking my Part A paper calculations after I did them by hand, and merging/compressing my scans into one PDF;
- writing the code for Part B and Part C. I made the design decisions: my fixed policy for Part B, the file structure, the tie rule and the seed scheme for Part C;
- helping me write the explanations in English. All results are from my own runs.

## Setup and reproduction

Python 3.12.7 on macOS. Gymnasium 1.4.0, NumPy 2.5.5 (see `requirements.txt`). Part C uses only standard Python.

Run from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python part_b/frozenlake.py
python part_c/solver.py
python part_c/simulate.py
```

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

Code: `part_c/gridworld.py` (map and move rules, read from `data/maps.json`), `part_c/solver.py` (value iteration and rule checks), `part_c/simulate.py` (episodes with the computed policy).

### Compare code with paper

| Position | Paper | Code |
| --- | --- | --- |
| V1 map | -1, -1, 10 / -1, #, -1 / -1, -1, -1, -1 | same |
| V2 (1,2) | 8, right | 8.0000, right |
| V2 (1,3) | 10, right | 10.0000, right |
| V2 (2,3) | 5.21, up | 5.2100, up |

No discrepancies. The code prints every action calculation, so it can be checked line by line against my paper work.

### Values, policies and convergence

Files: [values_paper.txt](outputs/part_c/values_paper.txt), [values_extended.txt](outputs/part_c/values_extended.txt)

| Map | Iterations | Final max change | Status |
| --- | --- | --- | --- |
| paper | 12 | 0.0000002165 | converged |
| extended | 11 | 0.0000002406 | converged |

Paper map values and policy:

```
 4.2000   8.0000  10.0000   G        →  →  →  G
 2.7800        #   5.8242   H        ↑  #  ↑  H
 1.5358   2.8176   4.2418  2.8176    →  →  ↑  ←
```

Extended map values and policy:

```
 4.2000   8.0000  10.0000   G        →  →  →  G
 2.7800        #   5.8242   H        ↑  #  ↑  H
 1.5020   2.8176   4.2418  2.8176    ↑  →  ↑  ←
```

Tie rule: actions whose Q value is within 1e-9 of the best are tied. The arrow is the first tied action in the order up, right, down, left. The code also lists all tied actions. In the final policies there were no ties.

### Simulation results

Files: [episodes.csv](outputs/part_c/episodes.csv), [summary.md](outputs/part_c/summary.md)

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| paper | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| paper | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

Each episode uses its own random generator with seed `1000 * batch_seed + i`. Start is S, step limit is 100.

### Explain your results

**What changed with sand and water.**
Only one state changed: S at (3,1). On the paper map, S goes right, then through the ice: S → (3,2) → (3,3) → ice → (1,3) → G.
On the extended map, the tile right of S is sand (-2). Going right now gives -2 + 0.9 × 2.8176 = 0.536. Going up gives 1.502. So S goes up and takes the path through mud: S → (2,1) → (1,1) → M → (1,3) → G. V(S) dropped from 1.5358 to 1.5020.
Water at (3,4) changed nothing. The policy never enters it, because entering costs -5. The value of W stays 2.8176, because from W the agent leaves to the left onto normal floor.

**Where cost and risk mattered.**
On the paper map, the ice path is shorter but risky. The mud path is safe but costs -3. The ice path was still a little better: 1.5358 against 1.502. Sand made the ice path more expensive, so the safe mud path won.

**How the policy handled ice.**
On ice the best action is up. With up, the slips go left (wall, stay on ice) or right (hole). The other actions are worse: right has 0.8 chance of the hole. The agent can still fall: the chance is 0.1 / 0.9 ≈ 11%, because a slip into the wall gives another try. In the simulation 14 of 150 paper episodes ended in the hole (9.3%), close to 11%.

**Simulations and values.**
V(S) is the expected discounted return over all possible slips, with no step limit. A simulation only gives a finite sample.
On the paper map the batches gave 0.89, 1.91 and 2.49, because the number of holes was different (8, 4, 2). The mean over 150 episodes is 1.7661, close to V(S) = 1.5358, but not equal. More episodes would bring it closer.
On the extended map the path has no ice, so every episode is the same and the mean is exactly V(S) = 1.5020.
The 100-step limit did not matter here: all episodes ended in 4–6 steps.

### Rule checks

File: [checks.txt](outputs/part_c/checks.txt). All 17 checks passed: boundary and wall moves stay in place with the tile reward, ice probabilities sum to 1 for every action, mud -3, sand -2, water -5, sand and water are deterministic, goal gives +10, G and H give no further rewards and keep value 0, the wall is not a state.