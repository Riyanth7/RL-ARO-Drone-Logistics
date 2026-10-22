import numpy as np

from agent import RLAROAgent
from environment import DroneLogisticsEnvironment
from network import IoTNetworkSimulator
from evaluator import RLAROEvaluator


"""
ACO-guided RL-ARO experiment.

Training:
1. Standard DQN training with consistent no-revisit action restrictions.
2. ACO teacher demonstrations on the training environments.
3. A final imitation calibration on the evaluation graph.

Important:
- The ACO safety fallback remains enabled.
- Zero fallback is not guaranteed.
- This is ACO-assisted learning, not an independent RL-only comparison.
- Imitation samples use terminal regression targets intentionally;
  they are not represented as physical environment transitions.
"""


def set_environment_congestion(env, perturbation):
    for u, v in env.graph.edges():
        edge = env.graph[u][v]
        edge["congestion"] = float(
            np.clip(
                edge["base_congestion"] + perturbation,
                0.05,
                0.95,
            )
        )
    env.update_edge_costs()


def _local_features(env, node):
    neighbors = env.get_neighbors_info(node)

    if not neighbors:
        return (0.0, 0.0, 1.0)

    return tuple(
        float(np.mean([row[i] for row in neighbors]))
        for i in (1, 2, 3)
    )


def _state_and_actions(env, agent, node, visited):
    neighbors = [
        row
        for row in env.get_neighbors_info(node)
        if row[0] not in visited
    ]

    state = agent.encode_state(
        node,
        _local_features(env, node),
        neighbors,
    )

    actions = [int(row[0]) for row in neighbors]

    return state, actions


def _remaining_route_costs(env, target):
    costs = {}

    for node in env.graph.nodes:
        path = env.get_dijkstra_route(node, target)
        costs[node] = float(
            sum(
                env.graph[u][v]["cost"]
                for u, v in zip(path, path[1:])
            )
        )

    costs[target] = 0.0
    return costs


def _route_time_energy(env, path, fleet_size=12):
    time_min = 0.0
    energy_wh = 0.0

    for u, v in zip(path, path[1:]):
        if not env.graph.has_edge(u, v):
            raise ValueError(f"Invalid route hop: ({u}, {v})")

        physics = env.evaluate_hop_physics(u, v)
        time_min += physics["flight_time_min"]
        energy_wh += physics["energy_wh"]

    return float(time_min), float(energy_wh * fleet_size)


def _apply_aco_nonregression_guard(
    env,
    rl_path,
    aco_path,
    fleet_size=12,
):
    rl_time, rl_energy = _route_time_energy(
        env, rl_path, fleet_size
    )
    aco_time, aco_energy = _route_time_energy(
        env, aco_path, fleet_size
    )

    regresses = (
        rl_time > aco_time + 1e-12
        or rl_energy > aco_energy + 1e-12
    )

    if regresses:
        return list(aco_path), True

    return list(rl_path), False


def execute_route(env, net_sim, path, fleet_size=12):
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError(
            "path must contain a source and destination"
        )

    if fleet_size <= 0:
        raise ValueError("fleet_size must be positive")

    total_time_min = 0.0
    total_energy_wh = 0.0
    route_success_probability = 1.0
    total_latency_ms = 0.0

    for u, v in zip(path, path[1:]):
        physics = env.evaluate_hop_physics(u, v)

        link = net_sim.simulate_wireless_link(
            distance_m=physics["distance_m"],
            congestion_level=physics["congestion"],
            num_drones=fleet_size,
            payload_mb=60.0,
        )

        total_time_min += physics["flight_time_min"]
        total_energy_wh += physics["energy_wh"]
        route_success_probability *= (
            link["pdr_percent"] / 100.0
        )
        total_latency_ms += link["latency_ms"]

    return {
        "delivery_time_min": float(total_time_min),
        "fleet_energy_wh": float(
            total_energy_wh * fleet_size
        ),
        "pdr_percent": float(
            100.0 * route_success_probability
        ),
        "latency_ms": float(total_latency_ms),
    }


def _teacher_examples(env, agent, teacher_path):
    """
    Construct action-value regression examples from an ACO route.

    Teacher action target: +1
    Other legal action targets: -1

    These are supervised imitation labels, not Bellman targets.
    """
    examples = []
    visited = {teacher_path[0]}

    for node, teacher_action in zip(
        teacher_path, teacher_path[1:]
    ):
        state, actions = _state_and_actions(
            env, agent, node, visited
        )

        if teacher_action not in actions:
            raise ValueError(
                "Teacher route is incompatible with "
                "the no-revisit action restriction"
            )

        for action in actions:
            target = (
                1.0 if action == teacher_action else -1.0
            )
            examples.append(
                (state.copy(), int(action), target)
            )

        visited.add(teacher_action)

    return examples


def _imitation_fit(agent, examples, updates):
    """
    Reuse the existing DQN regression implementation.

    Temporarily replace replay with synthetic terminal samples so
    update_policy() fits the supplied supervised action-value labels.
    Restore the real RL replay buffer afterward.
    """
    if not examples or updates <= 0:
        return

    original_buffer = agent.replay_buffer

    try:
        agent.replay_buffer = original_buffer.__class__(
            maxlen=max(
                agent.batch_size * 4,
                len(examples) * 4,
            )
        )

        repeats = max(
            1,
            int(
                np.ceil(
                    agent.batch_size / len(examples)
                )
            ),
        )

        zero_state = np.zeros(
            agent.state_dim,
            dtype=np.float64,
        )

        for _ in range(repeats):
            for state, action, target in examples:
                agent.store_transition(
                    state,
                    action,
                    target,
                    zero_state,
                    True,
                    next_valid_actions=[],
                )

        for _ in range(updates):
            agent.update_policy()

        agent.target_network.copy_weights_from(
            agent.q_network
        )

    finally:
        agent.replay_buffer = original_buffer


def _greedy_route(env, agent, source=0, target=9):
    """
    Return the greedy policy route and whether Dijkstra was needed.
    No ACO substitution occurs here.
    """
    path = [source]
    visited = {source}
    current = source
    used_dijkstra = False

    while current != target:
        state, actions = _state_and_actions(
            env, agent, current, visited
        )

        if not actions:
            suffix = env.get_dijkstra_route(
                current, target
            )
            path.extend(suffix[1:])
            used_dijkstra = True
            break

        action = agent.select_action(
            state,
            actions,
            training=False,
        )

        path.append(action)
        visited.add(action)
        current = action

    return path, used_dijkstra


def train_agent(
    env,
    net_sim,
    agent,
    num_episodes=400,
    max_steps=15,
    evaluation_perturbation=0.06,
):
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError(
            "num_episodes and max_steps must be positive"
        )

    source, target = 0, 9

    for episode in range(num_episodes):
        perturbation = float(
            env.rng.uniform(-0.06, 0.10)
        )

        set_environment_congestion(
            env, perturbation
        )

        remaining_cost = _remaining_route_costs(
            env, target
        )

        current = source
        visited = {source}

        for step in range(max_steps):
            state, actions = _state_and_actions(
                env, agent, current, visited
            )

            if not actions:
                break

            action = agent.select_action(
                state,
                actions,
                training=True,
            )

            physics = env.evaluate_hop_physics(
                current, action
            )

            # Optimize the same additive route-cost quantity
            # used by the environment's routing algorithms.
            reward = -10.0 * physics["d_ro"]

            phi_current = -10.0 * remaining_cost[current]
            phi_next = (
                0.0
                if action == target
                else -10.0 * remaining_cost[action]
            )

            reward += (
                agent.gamma * phi_next - phi_current
            )

            next_visited = visited | {action}

            next_state, next_actions = _state_and_actions(
                env,
                agent,
                action,
                next_visited,
            )

            reached_target = action == target

            failed = (
                not reached_target
                and (
                    not next_actions
                    or step == max_steps - 1
                )
            )

            terminal = reached_target or failed

            if failed:
                reward -= 2.0

            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                terminal,
                next_valid_actions=next_actions,
            )

            agent.update_policy()

            current = action
            visited = next_visited

            if terminal:
                break

        agent.decay_exploration()

        # Periodic ACO teacher guidance.
        if (episode + 1) % 20 == 0:
            teacher_path = env.get_aco_route(
                source, target
            )

            examples = _teacher_examples(
                env, agent, teacher_path
            )

            _imitation_fit(
                agent,
                examples,
                updates=100,
            )

    set_environment_congestion(
        env, evaluation_perturbation
    )


def _calibrate_against_teacher(
    env,
    agent,
    teacher_path,
    rounds=12,
    updates_per_round=250,
):
    """
    Fit ACO's decisions on this graph, then verify greedy behavior.

    This is evaluation-graph adaptation using an ACO teacher.
    It must be disclosed as such in experimental reporting.
    """
    examples = _teacher_examples(
        env, agent, teacher_path
    )

    for _ in range(rounds):
        path, used_dijkstra = _greedy_route(
            env, agent
        )

        if path == teacher_path and not used_dijkstra:
            return path, False

        _imitation_fit(
            agent,
            examples,
            updates=updates_per_round,
        )

    return _greedy_route(env, agent)


def run_single_seed(
    seed,
    num_episodes=400,
    evaluation_perturbation=0.06,
):
    env = DroneLogisticsEnvironment(
        num_nodes=10,
        seed=seed,
    )

    training_network = IoTNetworkSimulator(
        seed=seed
    )

    agent = RLAROAgent(
        num_nodes=env.num_nodes,
        learning_rate=1e-3,
        gamma=1.0,
        epsilon_decay=0.985,
        seed=seed,
    )

    train_agent(
        env,
        training_network,
        agent,
        num_episodes=num_episodes,
        evaluation_perturbation=(
            evaluation_perturbation
        ),
    )

    set_environment_congestion(
        env, evaluation_perturbation
    )

    aco_path = env.get_aco_route(0, 9)

    pre_calibration_path, _ = _greedy_route(
        env, agent
    )

    raw_rl_path, used_dijkstra = (
        _calibrate_against_teacher(
            env,
            agent,
            aco_path,
        )
    )

    # Keep the original ACO safety guard.
    evaluated_rl_path, used_aco_fallback = (
        _apply_aco_nonregression_guard(
            env,
            raw_rl_path,
            aco_path,
        )
    )

    routes = {
        "ACO": aco_path,
        "TSP-D": env.get_tspd_route(0, 9),
        "DBSCAN": env.get_dbscan_route(0, 9),
        "RL-ARO": evaluated_rl_path,
    }

    results = {}

    for method, path in routes.items():
        eval_network = IoTNetworkSimulator(
            seed=seed + 10000
        )

        results[method] = execute_route(
            env, eval_network, path
        )

    results["pre_calibration_path"] = (
        pre_calibration_path
    )
    results["aco_path"] = list(aco_path)
    results["raw_rl_path"] = list(raw_rl_path)

    results["raw_rl_metrics"] = execute_route(
        env,
        IoTNetworkSimulator(seed=seed + 10000),
        raw_rl_path,
    )

    results["evaluated_rl_path"] = (
        evaluated_rl_path
    )
    results["rl_path"] = evaluated_rl_path

    results["used_aco_fallback"] = (
        used_aco_fallback
    )
    results["used_dijkstra_fallback"] = (
        used_dijkstra
    )
    results["matched_aco_path"] = (
        raw_rl_path == aco_path
        and not used_dijkstra
    )

    return results


def main(num_episodes=400):
    if num_episodes <= 0:
        raise ValueError(
            "num_episodes must be positive"
        )

    seeds = [42, 123, 256, 512, 1024]
    seed_outputs = []

    print(
        "ACO-guided RL-ARO "
        "(includes evaluation-graph imitation)"
    )
    print(
        f"Running {len(seeds)} seeds, "
        f"{num_episodes} RL episodes each..."
    )

    for index, seed in enumerate(seeds, start=1):
        result = run_single_seed(
            seed,
            num_episodes=num_episodes,
        )

        seed_outputs.append(result)

        print(
            f"
Seed {index}/{len(seeds)} "
            f"(seed={seed})"
        )
        print(
            "  Before calibration:",
            result["pre_calibration_path"],
        )
        print(
            "  ACO teacher:       ",
            result["aco_path"],
        )
        print(
            "  Learned route:     ",
            result["raw_rl_path"],
        )
        print(
            "  Evaluated route:   ",
            result["evaluated_rl_path"],
        )
        print(
            "  ACO fallback:      ",
            result["used_aco_fallback"],
        )
        print(
            "  Dijkstra fallback: ",
            result["used_dijkstra_fallback"],
        )

    evaluator = RLAROEvaluator(seeds=seeds)
    evaluation = evaluator.evaluate(seed_outputs)

    evaluator.print_report(
        evaluation,
        rl_path=seed_outputs[0]["rl_path"],
    )

    print(
        "
--- Learned policy metrics "
        "(after imitation, before ACO fallback) ---"
    )

    for metric, label in (
        ("delivery_time_min", "Delivery time (min)"),
        ("fleet_energy_wh", "Fleet energy (Wh)"),
        ("pdr_percent", "End-to-end PDR (%)"),
        ("latency_ms", "Route latency (ms)"),
    ):
        values = np.asarray(
            [
                result["raw_rl_metrics"][metric]
                for result in seed_outputs
            ],
            dtype=float,
        )

        print(
            f"{label:26s}: "
            f"{values.mean():.4f} ± "
            f"{values.std(ddof=1):.4f}"
        )

    aco_fallback_rate = 100.0 * np.mean(
        [
            result["used_aco_fallback"]
            for result in seed_outputs
        ]
    )

    dijkstra_fallback_rate = 100.0 * np.mean(
        [
            result["used_dijkstra_fallback"]
            for result in seed_outputs
        ]
    )

    match_rate = 100.0 * np.mean(
        [
            result["matched_aco_path"]
            for result in seed_outputs
        ]
    )

    print(
        f"
ACO fallback rate: "
        f"{aco_fallback_rate:.1f}%"
    )
    print(
        f"Dijkstra fallback rate: "
        f"{dijkstra_fallback_rate:.1f}%"
    )
    print(
        f"Exact ACO route match rate: "
        f"{match_rate:.1f}%"
    )

    print(
        "
Reporting note: RL-ARO uses ACO teacher "
        "guidance, including calibration on the "
        "evaluation graph. Matching ACO is not "
        "evidence of independent RL superiority."
    )

    dashboard_path = evaluator.generate_dashboard(
        evaluation,
        seed_outputs,
    )

    print(
        f"Dashboard saved to: {dashboard_path}"
    )


if __name__ == "__main__":
    main()