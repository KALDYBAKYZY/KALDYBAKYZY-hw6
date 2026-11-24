from gridworld import ACTIONS, ARROWS, CONFIG, DISCOUNT, ROOT, extended_world, paper_world

OUTPUT_DIR = ROOT / "outputs" / "part_c"
TOLERANCE = CONFIG["solver"]["tolerance"]
MAX_ITERATIONS = CONFIG["solver"]["max_iterations"]
TIE_EPSILON = 1e-9
DISPLAY_EPSILON = 0.00005
TIE_RULE = (
    f"Tie rule: actions within {TIE_EPSILON} of the best Q value are tied; "
    "the arrow is the first tied action in the order up, right, down, left."
)


def q_value(world, values, state, action):
    return sum(
        probability * (reward + DISCOUNT * values[next_state])
        for probability, next_state, reward in world.outcomes(state, action)
    )


def bellman_update(world, values):
    return {
        state: 0.0 if world.is_terminal(state) else max(q_value(world, values, state, action) for action in ACTIONS)
        for state in world.states
    }


def value_iteration(world):
    values = {state: float(CONFIG["initial_value"]) for state in world.states}
    early = [values]
    change = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = bellman_update(world, values)
        change = max(abs(new_values[state] - values[state]) for state in world.states)
        values = new_values
        if len(early) < 3:
            early.append(values)
        if change < TOLERANCE:
            return values, iteration, change, True, early
    return values, MAX_ITERATIONS, change, False, early


def tied_actions(world, values, state):
    q_values = {action: q_value(world, values, state, action) for action in ACTIONS}
    best = max(q_values.values())
    return [action for action in ACTIONS if best - q_values[action] < TIE_EPSILON]


def extract_policy(world, values):
    ties = {
        state: tied_actions(world, values, state)
        for state in world.states
        if not world.is_terminal(state)
    }
    policy = {state: actions[0] for state, actions in ties.items()}
    return policy, ties


def value_map(world, values):
    lines = []
    for row in range(1, world.height + 1):
        cells = []
        for col in range(1, world.width + 1):
            state = (row, col)
            tile = world.tile(state)
            if state not in values or world.is_terminal(state):
                cells.append(f"{tile:>9}")
            else:
                cells.append(f"{values[state]:9.4f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def policy_map(world, policy):
    lines = []
    for row in range(1, world.height + 1):
        cells = []
        for col in range(1, world.width + 1):
            state = (row, col)
            cells.append(ARROWS[policy[state]] if state in policy else world.tile(state))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def ties_text(ties):
    return "\n".join(
        f"{state}: {', '.join(actions)}" + ("  (tie)" if len(actions) > 1 else "")
        for state, actions in ties.items()
    )


def v2_details(world, v1):
    lines = []
    for position in CONFIG["paper_v2_positions"]:
        state = tuple(position)
        lines.append(f"{state} tile {world.tile(state)}")
        q_values = {action: q_value(world, v1, state, action) for action in ACTIONS}
        for action, value in q_values.items():
            parts = " + ".join(
                f"{probability:g}*({reward} + {DISCOUNT}*{v1[next_state]:g})"
                for probability, next_state, reward in world.outcomes(state, action)
            )
            lines.append(f"  {action:>5}: {parts} = {value:.4f}")
        best = max(q_values.values())
        best_actions = [action for action in ACTIONS if best - q_values[action] < TIE_EPSILON]
        lines.append(f"  V2 = {best:.4f}, best: {', '.join(best_actions)}")
    return "\n".join(lines)


def convergence_text(iterations, change, converged):
    status = "converged" if converged else f"NOT converged: reached {MAX_ITERATIONS} iterations"
    return f"Iterations: {iterations}\nFinal max change: {change:.10f}\nStatus: {status}"


def solve(world):
    values, iterations, change, converged, early = value_iteration(world)
    policy, ties = extract_policy(world, values)
    return {
        "values": values,
        "iterations": iterations,
        "change": change,
        "converged": converged,
        "early": early,
        "policy": policy,
        "ties": ties,
    }


def report(world, result):
    sections = [
        f"Map: {world.name}",
        "\n".join(world.rows),
        "",
        "Convergence",
        convergence_text(result["iterations"], result["change"], result["converged"]),
        "",
        "Final values",
        value_map(world, result["values"]),
        "",
        "Policy",
        policy_map(world, result["policy"]),
        "",
        "Best actions per state",
        ties_text(result["ties"]),
        "",
        TIE_RULE,
    ]
    return "\n".join(sections)


def paper_report(world, result):
    v1 = result["early"][1]
    sections = [
        "V1",
        value_map(world, v1),
        "",
        "V2 at requested positions (using V1)",
        v2_details(world, v1),
        "",
        report(world, result),
    ]
    return "\n".join(sections)


def changes_text(paper, paper_result, extended, extended_result):
    lines = ["Changes compared with the paper map (state: paper -> extended)"]
    for state in extended.states:
        if extended.is_terminal(state):
            continue
        old_value = paper_result["values"][state]
        new_value = extended_result["values"][state]
        old_action = paper_result["policy"][state]
        new_action = extended_result["policy"][state]
        if abs(old_value - new_value) > DISPLAY_EPSILON or old_action != new_action:
            lines.append(
                f"{state} {paper.tile(state)}->{extended.tile(state)}: "
                f"value {old_value:.4f} -> {new_value:.4f}, action {old_action} -> {new_action}"
            )
    return "\n".join(lines)


def check(description, passed):
    return f"[{'PASS' if passed else 'FAIL'}] {description}"


def run_checks(paper, extended, paper_result):
    lines = [
        check("Boundary: (1,1) up stays at (1,1) with reward -1", paper.outcomes((1, 1), "up") == [(1.0, (1, 1), -1)]),
        check("Wall: (3,2) up stays at (3,2) with reward -1", paper.outcomes((3, 2), "up") == [(1.0, (3, 2), -1)]),
        check("Wall on ice: (2,3) left stays on ice with probability 0.8", paper.outcomes((2, 3), "left")[0] == (0.8, (2, 3), -1)),
        check("Mud blocked: (1,2) up stays on mud with reward -3", paper.outcomes((1, 2), "up") == [(1.0, (1, 2), -3)]),
    ]
    for action in ACTIONS:
        total = sum(probability for probability, _, _ in paper.outcomes((2, 3), action))
        lines.append(check(f"Ice probabilities for {action} sum to 1 (sum = {total:g})", abs(total - 1) < 1e-12))
    lines += [
        check("Mud reward: (1,1) right into M gives -3", paper.outcomes((1, 1), "right") == [(1.0, (1, 2), -3)]),
        check("Sand reward: (3,1) right into D gives -2", extended.outcomes((3, 1), "right") == [(1.0, (3, 2), -2)]),
        check("Water reward: (3,3) right into W gives -5", extended.outcomes((3, 3), "right") == [(1.0, (3, 4), -5)]),
        check("Sand is deterministic: one outcome from (3,2)", all(len(extended.outcomes((3, 2), a)) == 1 for a in ACTIONS)),
        check("Water is deterministic: one outcome from (3,4)", all(len(extended.outcomes((3, 4), a)) == 1 for a in ACTIONS)),
        check("Goal gives +10 on entry: (1,3) right", paper.outcomes((1, 3), "right") == [(1.0, (1, 4), 10)]),
        check("No rewards after termination: G and H have no outcomes", all(paper.outcomes(s, a) == [] for s in [(1, 4), (2, 4)] for a in ACTIONS)),
        check("Terminal values stay 0", paper_result["values"][(1, 4)] == 0 and paper_result["values"][(2, 4)] == 0),
        check("Wall is not a state", (2, 2) not in paper.states),
    ]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = paper_world()
    extended = extended_world()
    paper_result = solve(paper)
    extended_result = solve(extended)

    checks = run_checks(paper, extended, paper_result)
    paper_text = paper_report(paper, paper_result)
    extended_text = report(extended, extended_result) + "\n\n" + changes_text(paper, paper_result, extended, extended_result)

    (OUTPUT_DIR / "checks.txt").write_text(checks + "\n")
    (OUTPUT_DIR / "values_paper.txt").write_text(paper_text + "\n")
    (OUTPUT_DIR / "values_extended.txt").write_text(extended_text + "\n")

    print("RULE CHECKS\n" + checks)
    print("\n" + "=" * 60 + "\n" + paper_text)
    print("\n" + "=" * 60 + "\n" + extended_text)


if __name__ == "__main__":
    main()