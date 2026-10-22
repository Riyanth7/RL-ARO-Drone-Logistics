
import numpy as np

from agent import RLAROAgent
from environment import DroneLogisticsEnvironment
from network import IoTNetworkSimulator
from evaluator import RLAROEvaluator


# ================================================================
# Environment helpers
# ================================================================

def set_environment_congestion(env, perturbation):
    """
    Set congestion relative to each edge's base congestion.

    The same perturbation is applied to every edge so that
    experiments use a consistent environmental condition.
    """
    if not np.isfinite(perturbation):
        raise ValueError("perturbation must be finite")

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
    """
    Return the mean local energy, congestion, and QoS features.

    Feature order:
        (energy_norm, congestion, qos)
    """
    neighbors = env.get_neighbors_info(node)

    if not neighbors:
        return (0.0, 0.0, 1.0)

    return tuple(
        float(np.mean([row[i] for row in neighbors]))
        for i in (1, 2, 3)
    )


# ================================================================
# Route evaluation
# ================================================================

def execute_route(env, net_sim, path, fleet_size=1):
    """
    Evaluate a route using flight physics and wireless-link metrics.

    This experiment uses one drone. fleet_size remains an argument
    to preserve compatibility with the previous implementation.
    """
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError(
            "path must contain at least a source and destination"
        )

    if fleet_size != 1:
        raise ValueError("This experiment is configured for one drone")

    total_time_min = 0.0
    total_energy_wh = 0.0
    route_success_probability = 1.0
    total_network_latency_ms = 0.0

    for u, v in zip(path, path[1:]):
        if not env.graph.has_edge(u, v):
            raise ValueError(
                f"Route contains non-edge hop ({u}, {v})"
            )

        physics = env.evaluate_hop_physics(u, v)

        link = net_sim.simulate_wireless_link(
            distance_m=physics["distance_m"],
            congestion_level=physics["congestion"],
            num_drones=1,
            payload_mb=60.0,
        )

        total_time_min += physics["flight_time_min"]
        total_energy_wh += physics["energy_wh"]

        route_success_probability *= (
            link["pdr_percent"] / 100.0
        )

        total_network_latency_ms += link["latency_ms"]

    return {
        "delivery_time_min": float(total_time_min),
        "fleet_energy_wh": float(total_energy_wh),
        "pdr_percent": float(100.0 * route_success_probability),
        "latency_ms": float(total_network_latency_ms),
    }


def _remaining_route_costs(env, target):
    """Calculate the minimum additive edge cost from every node."""
    costs = {}

    # Refresh once before calculating the remaining route costs.
    env.update_edge_costs()

    for node in env.graph.nodes:
        path = env.get_dijkstra_route(node, target)

        if not path or path[-1] != target:
            raise ValueError(
                f"No valid route from node {node} to target {target}"
            )

        costs[node] = float(
            sum(
                env.graph[u][v]["cost"]
                for u, v in zip(path, path[1:])
            )
        )

    costs[target] = 0.0
    return costs


def _route_time_energy(env, path, fleet_size=1):
    """Calculate deterministic flight time and energy for a route."""
    if not path:
        raise ValueError("path cannot be empty")

    if fleet_size != 1:
        raise ValueError("This experiment is configured for one drone")

    total_time_min = 0.0
    total_energy_wh = 0.0

    for u, v in zip(path, path[1:]):
        if not env.graph.has_edge(u, v):
            raise ValueError(
                f"Route contains non-edge hop ({u}, {v})"
            )

        physics = env.evaluate_hop_physics(u, v)
        total_time_min += physics["flight_time_min"]
        total_energy_wh += physics["energy_wh"]

    return float(total_time_min), float(total_energy_wh)


def _apply_aco_nonregression_guard(
    env, rl_path, aco_path, fleet_size=1
):
    """
    Use ACO if the raw RL route worsens delivery time or energy.

    This guard evaluates a hybrid system; it does not prove that
    the raw RL policy outperforms ACO.
    """
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


# ================================================================
# Training
# ================================================================

def train_agent(
    env,
    net_sim,
    agent,
    num_episodes=300,
    max_steps=15,
    evaluation_perturbation=0.06,
):
    """
    Train the RL-ARO agent and return its greedy route.

    The first 80% of episodes use randomized congestion.
    The remaining episodes use the fixed evaluation perturbation.
    """
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError(
            "num_episodes and max_steps must be positive"
        )

    if not np.isfinite(evaluation_perturbation):
        raise ValueError("evaluation_perturbation must be finite")

    source, target = 0, 9

    for episode in range(num_episodes):
        if episode < int(num_episodes * 0.8):
            episode_perturbation = float(
                env.rng.uniform(-0.06, 0.10)
            )
        else:
            episode_perturbation = float(
                evaluation_perturbation
            )

        set_environment_congestion(
            env, episode_perturbation
        )

        remaining_cost = _remaining_route_costs(env, target)

        current = source
        visited = {source}

        for _step in range(max_steps):
            if current == target:
                break

            # Only unvisited neighbors are legal actions.
            neighbor_info = [
                item
                for item in env.get_neighbors_info(current)
                if item[0] not in visited
            ]

            if not neighbor_info:
                # The previous transition into this dead end should
                # already have been marked terminal and penalized.
                break

            valid_actions = [
                int(item[0]) for item in neighbor_info
            ]

            state = agent.encode_state(
                current,
                _local_features(env, current),
                neighbor_info,
                target_node=target,
            )

            action = agent.select_action(
                state,
                valid_actions,
                training=True,
            )

            physics = env.evaluate_hop_physics(
                current, action
            )

            link = net_sim.simulate_wireless_link(
                distance_m=physics["distance_m"],
                congestion_level=physics["congestion"],
                num_drones=1,
            )

            reward = agent.compute_reward(
                energy_saved_norm=np.clip(
                    physics["energy_saved_wh"] / env.e_max_wh,
                    0.0,
                    1.0,
                ),
                energy_consumed_norm=np.clip(
                    physics["energy_wh"] / env.e_max_wh,
                    0.0,
                    1.0,
                ),
                data_loss_rate=link["packet_loss"],
                route_cost_dro=physics["d_ro"],
                qos_index=link["q_svc"],
                auth_delay_ms=link["b_auth_ms"],
            )

            # Potential-based shaping using remaining additive d_ro cost.
            phi_current = -remaining_cost[current]
            phi_next = (
                0.0
                if action == target
                else -remaining_cost[action]
            )

            reward += 2.5 * (
                agent.gamma * phi_next - phi_current
            )

            done = action == target

            if done:
                reward += 3.0

            visited.add(action)

            # Keep the same neighbor-filtering convention when
            # constructing the next state and its legal actions.
            next_neighbors = [
                item
                for item in env.get_neighbors_info(action)
                if item[0] not in visited
            ]

            next_valid_actions = [
                int(item[0]) for item in next_neighbors
            ]

            dead_end = not done and not next_valid_actions

            if dead_end:
                # Explicitly discourage routes that strand the agent.
                reward -= 3.0

            transition_done = done or dead_end

            next_state = agent.encode_state(
                action,
                _local_features(env, action),
                next_neighbors,
                target_node=target,
            )

            agent.store_transition(
                state,
                action,
                reward,
                next_state,
                transition_done,
                next_valid_actions=next_valid_actions,
            )

            agent.update_policy()

            current = action

            if transition_done:
                break

        # Matches the refined RLAROAgent method name.
        agent.decay_epsilon()

    # ============================================================
    # Greedy evaluation of the learned policy
    # ============================================================

    set_environment_congestion(
        env, evaluation_perturbation
    )

    path = [source]
    current = source
    visited = {source}

    while current != target and len(path) < env.num_nodes:
        neighbor_info = [
            item
            for item in env.get_neighbors_info(current)
            if item[0] not in visited
        ]

        if not neighbor_info:
            # Complete the route using Dijkstra if the learned policy
            # reaches a dead end. This completion is not a pure RL route.
            remainder = env.get_dijkstra_route(current, target)
            path.extend(remainder[1:])
            break

        state = agent.encode_state(
            current,
            _local_features(env, current),
            neighbor_info,
            target_node=target,
        )

        next_node = agent.select_action(
            state,
            [int(item[0]) for item in neighbor_info],
            training=False,
        )

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    if path[-1] != target:
        remainder = env.get_dijkstra_route(path[-1], target)
        path.extend(remainder[1:])

    return path


# ================================================================
# One seed
# ================================================================

def run_single_seed(
    seed,
    num_episodes=300,
    evaluation_perturbation=0.06,
):
    """Train and evaluate all methods for one seed."""
    env = DroneLogisticsEnvironment(
        num_nodes=10,
        seed=seed,
    )

    training_network = IoTNetworkSimulator(seed=seed)

    agent = RLAROAgent(
        num_nodes=env.num_nodes,
        learning_rate=5e-4,
        epsilon_decay=0.992,
        target_update_freq=100,
        seed=seed,
    )

    raw_rl_path = train_agent(
        env,
        training_network,
        agent,
        num_episodes=num_episodes,
        evaluation_perturbation=evaluation_perturbation,
    )

    # All methods must be evaluated in the same environmental condition.
    set_environment_congestion(
        env, evaluation_perturbation
    )

    aco_path = env.get_aco_route(0, 9)

    evaluated_rl_path, used_aco_fallback = (
        _apply_aco_nonregression_guard(
            env,
            raw_rl_path,
            aco_path,
            fleet_size=1,
        )
    )

    routes = {
        "ACO": aco_path,
        "TSP-D": env.get_tspd_route(0, 9),
        "DBSCAN": env.get_dbscan_route(0, 9),
        "RL-ARO": evaluated_rl_path,
    }

    results = {}

    # Use the same simulator seed for each method within this trial.
    # This improves comparability, although different route lengths
    # can still produce different random-number consumption.
    for method, path in routes.items():
        eval_network = IoTNetworkSimulator(
            seed=seed + 10000
        )
        results[method] = execute_route(
            env,
            eval_network,
            path,
            fleet_size=1,
        )

    raw_network = IoTNetworkSimulator(
        seed=seed + 10000
    )

    results["raw_rl_path"] = list(raw_rl_path)
    results["raw_rl_metrics"] = execute_route(
        env,
        raw_network,
        raw_rl_path,
        fleet_size=1,
    )
    results["evaluated_rl_path"] = list(evaluated_rl_path)
    results["rl_path"] = list(evaluated_rl_path)
    results["aco_path"] = list(aco_path)
    results["used_aco_fallback"] = bool(
        used_aco_fallback
    )

    return results


# ================================================================
# Experiment driver
# ================================================================

def main(num_episodes=300):
    """Run the experiment using five reproducible seeds."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    seeds = [42, 123, 256, 512, 1024]

    print("=" * 68)
    print("RL-ARO DRONE ROUTING EXPERIMENT")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {num_episodes}")
    print("Drone count: 1")
    print("=" * 68)

    seed_outputs = []

    for index, seed in enumerate(seeds, start=1):
        result = run_single_seed(
            seed,
            num_episodes=num_episodes,
        )

        seed_outputs.append(result)

        print(f"\nSeed {index}/{len(seeds)}: {seed}")
        print(f"  Raw RL route: {result['raw_rl_path']}")
        print(f"  ACO route:    {result['aco_path']}")
        print(
            "  Evaluated route: "
            f"{result['evaluated_rl_path']}"
        )
        print(
            "  ACO fallback: "
            f"{result['used_aco_fallback']}"
        )

        raw_metrics = result["raw_rl_metrics"]
        print(
            "  Raw RL time: "
            f"{raw_metrics['delivery_time_min']:.3f} min"
        )
        print(
            "  Raw RL energy: "
            f"{raw_metrics['fleet_energy_wh']:.3f} Wh"
        )

    # Existing evaluator interface retained.
    evaluator = RLAROEvaluator(seeds=seeds)
    evaluation = evaluator.evaluate(seed_outputs)

    evaluator.print_report(
        evaluation,
        rl_path=seed_outputs[0]["rl_path"],
    )

    # Report the raw RL policy separately from the hybrid result.
    print("\n" + "=" * 68)
    print("RAW RL-ARO POLICY METRICS (BEFORE ACO FALLBACK)")
    print("=" * 68)

    metrics_to_report = [
        ("delivery_time_min", "Delivery time (min)"),
        ("fleet_energy_wh", "Fleet energy (Wh)"),
        ("pdr_percent", "End-to-end PDR (%)"),
        ("latency_ms", "Route latency (ms)"),
    ]

    for metric, label in metrics_to_report:
        values = np.asarray(
            [
                result["raw_rl_metrics"][metric]
                for result in seed_outputs
            ],
            dtype=np.float64,
        )

        print(
            f"{label:28s}: "
            f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"
        )

    fallback_count = sum(
        result["used_aco_fallback"]
        for result in seed_outputs
    )
    fallback_rate = 100.0 * fallback_count / len(seed_outputs)

    print(
        f"\nACO fallback count: {fallback_count}/{len(seeds)}"
    )
    print(f"ACO fallback rate: {fallback_rate:.1f}%")

    dashboard_path = evaluator.generate_dashboard(
        evaluation,
        seed_outputs,
    )
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main(num_episodes=300)
