import csv
import re
from pathlib import Path

import gymnasium as gym
import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
GOAL_STATE = 15
MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "part_b"

FIXED_POLICY = {
    0: DOWN,
    1: LEFT,
    2: DOWN,
    3: LEFT,
    4: DOWN,
    6: DOWN,
    8: RIGHT,
    9: DOWN,
    10: DOWN,
    13: RIGHT,
    14: RIGHT,
}


def make_env(slippery):
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=slippery,
        max_episode_steps=MAX_STEPS,
        render_mode="ansi",
    )


def choose_action(policy, state, rng):
    if policy == "random":
        return int(rng.integers(4))
    return FIXED_POLICY[state]


def run_episode(env, policy, episode_seed, rng):
    state, _ = env.reset(seed=episode_seed)
    frames = [env.render()]
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = choose_action(policy, state, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        frames.append(env.render())
    if terminated:
        outcome = "goal" if state == GOAL_STATE else "hole"
    else:
        outcome = "time limit"
    return steps, total_reward, outcome, frames


def run_all():
    rows = []
    traces = {}
    for slippery in (False, True):
        env = make_env(slippery)
        for policy in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'not_slippery'}_{policy}"
            for batch_seed in BATCH_SEEDS:
                rng = np.random.default_rng(batch_seed)
                for i in range(EPISODES_PER_BATCH):
                    episode_seed = 1000 * batch_seed + i
                    steps, total_reward, outcome, frames = run_episode(env, policy, episode_seed, rng)
                    rows.append({
                        "condition": condition,
                        "slippery": slippery,
                        "batch_seed": batch_seed,
                        "episode_seed": episode_seed,
                        "policy": policy,
                        "steps": steps,
                        "total_reward": total_reward,
                        "outcome": outcome,
                    })
                    traces.setdefault((condition, outcome), (episode_seed, steps, frames))
        env.close()
    return rows, traces


def save_episodes(rows):
    with open(OUTPUT_DIR / "episodes.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows):
    lines = [
        f"Gymnasium {gym.__version__}, NumPy {np.__version__}",
        "",
        "| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for slippery in (False, True):
        for policy in ("random", "fixed"):
            for batch_seed in BATCH_SEEDS:
                batch = [
                    row for row in rows
                    if row["slippery"] == slippery and row["policy"] == policy and row["batch_seed"] == batch_seed
                ]
                goals = sum(row["outcome"] == "goal" for row in batch)
                holes = sum(row["outcome"] == "hole" for row in batch)
                time_limits = sum(row["outcome"] == "time limit" for row in batch)
                success = 100 * goals / len(batch)
                mean_steps = sum(row["steps"] for row in batch) / len(batch)
                lines.append(
                    f"| {'yes' if slippery else 'no'} | {policy} | {batch_seed} | {goals} | {holes} | "
                    f"{time_limits} | {success:.0f}% | {mean_steps:.2f} |"
                )
    return "\n".join(lines)


def clean_frame(frame):
    return re.sub(r"\x1b\[41m.\x1b\[0m", "*", frame)


def format_trace(title, trace):
    if trace is None:
        return f"=== {title} ===\nNot observed.\n"
    episode_seed, steps, frames = trace
    return f"=== {title} | episode seed {episode_seed} | {steps} steps ===\n" + "\n".join(clean_frame(frame) for frame in frames) + "\n"


def save_traces(traces):
    failure = traces.get(("slippery_fixed", "hole")) or traces.get(("slippery_fixed", "time limit"))
    parts = [
        "* = agent position",
        "",
        format_trace("Example episode: not slippery, fixed policy", traces.get(("not_slippery_fixed", "goal"))),
        format_trace("Successful episode: slippery, fixed policy", traces.get(("slippery_fixed", "goal"))),
        format_trace("Failed episode: slippery, fixed policy", failure),
    ]
    (OUTPUT_DIR / "traces.txt").write_text("\n".join(parts))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, traces = run_all()
    save_episodes(rows)
    summary = make_summary(rows)
    (OUTPUT_DIR / "summary.md").write_text(summary + "\n")
    save_traces(traces)
    print(summary)


if __name__ == "__main__":
    main()