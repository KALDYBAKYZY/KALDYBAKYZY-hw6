import csv
import random

from gridworld import CONFIG, DISCOUNT, ROOT, extended_world, paper_world
from solver import solve

OUTPUT_DIR = ROOT / "outputs" / "part_c"
SIMULATION = CONFIG["simulation"]
BATCH_SEEDS = SIMULATION["batch_seeds"]
EPISODES_PER_BATCH = SIMULATION["episodes_per_batch"]
MAX_STEPS = SIMULATION["max_steps"]


def sample_outcome(world, state, action, rng):
    outcomes = world.outcomes(state, action)
    roll = rng.random()
    cumulative = 0.0
    for probability, next_state, reward in outcomes:
        cumulative += probability
        if roll < cumulative:
            return next_state, reward
    return outcomes[-1][1], outcomes[-1][2]


def run_episode(world, policy, episode_seed):
    rng = random.Random(episode_seed)
    state = world.start()
    steps = 0
    total_reward = 0.0
    discounted_return = 0.0
    path = [state]
    while steps < MAX_STEPS and not world.is_terminal(state):
        state, reward = sample_outcome(world, state, policy[state], rng)
        discounted_return += DISCOUNT ** steps * reward
        total_reward += reward
        steps += 1
        path.append(state)
    if world.tile(state) == "G":
        outcome = "goal"
    elif world.tile(state) == "H":
        outcome = "hole"
    else:
        outcome = "time limit"
    return steps, total_reward, discounted_return, outcome, path


def simulate(world, policy):
    rows = []
    for batch_seed in BATCH_SEEDS:
        for i in range(EPISODES_PER_BATCH):
            episode_seed = 1000 * batch_seed + i
            steps, total_reward, discounted_return, outcome, path = run_episode(world, policy, episode_seed)
            rows.append({
                "map": world.name,
                "batch_seed": batch_seed,
                "episode_seed": episode_seed,
                "steps": steps,
                "total_reward": total_reward,
                "discounted_return": round(discounted_return, 4),
                "outcome": outcome,
                "path": " ".join(f"({row},{col})" for row, col in path),
            })
    return rows


def summary_lines(world, rows, start_value):
    lines = []
    for batch_seed in BATCH_SEEDS:
        batch = [row for row in rows if row["batch_seed"] == batch_seed]
        goals = sum(row["outcome"] == "goal" for row in batch)
        holes = sum(row["outcome"] == "hole" for row in batch)
        time_limits = sum(row["outcome"] == "time limit" for row in batch)
        mean_steps = sum(row["steps"] for row in batch) / len(batch)
        mean_return = sum(row["discounted_return"] for row in batch) / len(batch)
        lines.append(
            f"| {world.name} | {batch_seed} | {goals} | {holes} | {time_limits} | "
            f"{mean_steps:.2f} | {mean_return:.4f} |"
        )
    all_return = sum(row["discounted_return"] for row in rows) / len(rows)
    note = f"{world.name}: V(S) = {start_value:.4f}, mean discounted return over all {len(rows)} episodes = {all_return:.4f}"
    return lines, note


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    table = [
        "| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    notes = []
    for world in (paper_world(), extended_world()):
        result = solve(world)
        rows = simulate(world, result["policy"])
        lines, note = summary_lines(world, rows, result["values"][world.start()])
        table += lines
        notes.append(note)
        all_rows += rows

    with open(OUTPUT_DIR / "episodes.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summary = "\n".join(table) + "\n\n" + "\n".join(notes)
    (OUTPUT_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()