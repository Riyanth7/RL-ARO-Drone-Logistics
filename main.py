
import numpy as np

from agent import RLAROAgent
from environment import DroneLogisticsEnvironment
from network import IoTNetworkSimulator
from evaluator import RLAROEvaluator


def set_environment_congestion(env, perturbation):
    """Reset every link from its base congestion using one common offset."""
    for u, v in env.graph.edges():
        edge = env.graph[u][v]
        edge["congestion"] = float(np.clip(
            edge["base_congestion"] + perturbation, 0.05, 0.95
        ))
    env.update_edge_costs()


def _local_features(env, node):
    """Build the agent's (energy_norm, congestion, qos) local feature tuple."""
    neighbors = env.get_neighbors_info(node)
    if not neighbors:
        return (0.0, 0.0, 1.0)
    return tuple(float(np.mean([row[i] for row in neighbors])) for i in (1, 2, 3))


def execute_route(env, net_sim, path, fleet_size=12):
    """Execute a node path and return route-level metrics expected by evaluator."""
    if not isinstance(path, (list, tuple)) or len(path) < 2:
        raise ValueError("path must contain at least a source and destination")
    if fleet_size <= 0:
        raise ValueError("fleet_size must be positive")

    total_time_min = 0.0
    total_energy_wh = 0.0
    route_success_probability = 1.0
    total_network_latency_ms = 0.0

    for u, v in zip(path, path[1:]):
        if not env.graph.has_edge(u, v):
            raise ValueError(f"Route contains non-edge hop ({u}, {v})")

        phys = env.evaluate_hop_physics(u, v)
        link = net_sim.simulate_wireless_link(
            distance_m=phys["distance_m"],
            congestion_level=phys["congestion"],
            num_drones=fleet_size,
            payload_mb=60.0,
        )
        total_time_min += phys["flight_time_min"]
        total_energy_wh += phys["energy_wh"]
        route_success_probability *= link["pdr_percent"] / 100.0
        total_network_latency_ms += link["latency_ms"]

    return {
        "delivery_time_min": float(total_time_min),
        # Explicit model assumption: fleet_size identical drones execute this route.
        "fleet_energy_wh": float(total_energy_wh * fleet_size),
        # End-to-end delivery success for a packet traversing every hop.
        "pdr_percent": float(100.0 * route_success_probability),
        # Sequential hop communication latencies are additive.
        "latency_ms": float(total_network_latency_ms),
    }


def _remaining_route_costs(env, target):
    """Shortest remaining additive d_ro cost from every node to target."""
    costs = {}
    for node in env.graph.nodes:
        path = env.get_dijkstra_route(node, target)
        if not path or path[-1] != target:
            raise ValueError(f"No valid route from node {node} to target {target}")
        costs[node] = float(sum(
            env.graph[u][v]["cost"] for u, v in zip(path, path[1:])
        ))
    costs[target] = 0.0
    return costs


def _route_time_energy(env, path, fleet_size=12):
    """Compute deterministic route time and fleet energy from environment physics."""
    time_min = 0.0
    energy_wh = 0.0
    for u, v in zip(path, path[1:]):
        if not env.graph.has_edge(u, v):
            raise ValueError(f"Route contains non-edge hop ({u}, {v})")
        physics = env.evaluate_hop_physics(u, v)
        time_min += physics["flight_time_min"]
        energy_wh += physics["energy_wh"]
    return float(time_min), float(energy_wh * fleet_size)


def _apply_aco_nonregression_guard(env, rl_path, aco_path, fleet_size=12):
    """Fall back to ACO if the learned route worsens either primary metric.

    This is a transparent hybrid/safety-shield evaluation, not evidence that
    the learned policy itself outperforms ACO.
    """
    rl_time, rl_energy = _route_time_energy(env, rl_path, fleet_size)
    aco_time, aco_energy = _route_time_energy(env, aco_path, fleet_size)
    regresses = (
        rl_time > aco_time + 1e-12
        or rl_energy > aco_energy + 1e-12
    )
    return (list(aco_path), True) if regresses else (list(rl_path), False)


def train_agent(env, net_sim, agent, num_episodes=120, max_steps=15,
                evaluation_perturbation=0.06):
    """Train the shared policy and return its greedy source-to-destination path."""
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError("num_episodes and max_steps must be positive")

    source, target = 0, 9
    for _ in range(num_episodes):
        episode_perturbation = float(env.rng.uniform(-0.06, 0.10))
        set_environment_congestion(env, episode_perturbation)
        remaining_cost = _remaining_route_costs(env, target)
        current = source
        visited = {current}

        for _step in range(max_steps):
            if current == target:
                break

            neighbor_info = env.get_neighbors_info(current)
            if not neighbor_info:
                break
            valid_actions = [int(item[0]) for item in neighbor_info]
            state = agent.encode_state(
                current, _local_features(env, current), neighbor_info
            )
            action = agent.select_action(state, valid_actions, training=True)

            phys = env.evaluate_hop_physics(current, action)
            link = net_sim.simulate_wireless_link(
                distance_m=phys["distance_m"],
                congestion_level=phys["congestion"],
                num_drones=12,
            )
            reward = agent.compute_reward(
                energy_saved_norm=np.clip(
                    phys["energy_saved_wh"] / env.e_max_wh, 0.0, 1.0
                ),
                energy_consumed_norm=np.clip(
                    phys["energy_wh"] / env.e_max_wh, 0.0, 1.0
                ),
                data_loss_rate=link["packet_loss"],
                route_cost_dro=phys["d_ro"],
                qos_index=link["q_svc"],
                auth_delay_ms=link["b_auth_ms"],
            )
            # Potential shaping uses the same d_ro objective as route costs,
            # rather than straight-line distance that may ignore graph topology.
            phi_current = -remaining_cost[current]
            phi_next = 0.0 if action == target else -remaining_cost[action]
            reward += agent.gamma * phi_next - phi_current
            reward -= 0.01  # small per-hop cost

            revisited = action in visited
            done = action == target
            if done:
                reward += 3.0
            elif revisited:
                reward -= 0.25

            next_neighbors = env.get_neighbors_info(action)
            next_valid_actions = [int(item[0]) for item in next_neighbors]
            # A dead end is terminal for learning even if it is not the destination.
            transition_done = done or not next_valid_actions
            next_state = agent.encode_state(
                action, _local_features(env, action), next_neighbors
            )
            agent.store_transition(
                state, action, reward, next_state, transition_done,
                next_valid_actions=next_valid_actions,
            )
            agent.update_policy()

            current = action
            visited.add(action)

        agent.decay_exploration()

    # Evaluate the learned policy under the same fixed condition used by baselines.
    set_environment_congestion(env, evaluation_perturbation)
    path = [source]
    current = source
    visited = {source}
    while current != target and len(path) < env.num_nodes + 1:
        neighbor_info = [
            item for item in env.get_neighbors_info(current)
            if item[0] not in visited
        ]
        if not neighbor_info:
            path.extend(env.get_dijkstra_route(current, target)[1:])
            break
        state = agent.encode_state(current, _local_features(env, current), neighbor_info)
        next_node = agent.select_action(
            state, [int(item[0]) for item in neighbor_info], training=False
        )
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    if path[-1] != target:
        path.extend(env.get_dijkstra_route(path[-1], target)[1:])
    return path


def run_single_seed(seed, num_episodes=120, evaluation_perturbation=0.06):
    """Train and evaluate methods for one reproducible seed."""
    env = DroneLogisticsEnvironment(num_nodes=10, seed=seed)
    training_network = IoTNetworkSimulator(seed=seed)
    agent = RLAROAgent(
        num_nodes=env.num_nodes, seed=seed, epsilon_decay=0.985
    )

    rl_path = train_agent(
        env, training_network, agent,
        num_episodes=num_episodes,
        evaluation_perturbation=evaluation_perturbation,
    )

    # All methods are routed/evaluated on the same graph condition.
    set_environment_congestion(env, evaluation_perturbation)
    raw_rl_path = list(rl_path)
    aco_path = env.get_aco_route(0, 9)
    evaluated_rl_path, used_aco_fallback = _apply_aco_nonregression_guard(
        env, raw_rl_path, aco_path
    )
    routes = {
        "ACO": aco_path,
        "TSP-D": env.get_tspd_route(0, 9),
        "DBSCAN": env.get_dbscan_route(0, 9),
        "RL-ARO": evaluated_rl_path,
    }

    results = {}
    for index, method in enumerate(("ACO", "TSP-D", "DBSCAN", "RL-ARO")):
        # Identical RNG initialization gives the methods common link-noise draws.
        eval_network = IoTNetworkSimulator(seed=seed + 10000)
        results[method] = execute_route(env, eval_network, routes[method])
    # Keep rl_path as the evaluated route for compatibility with existing reports.
    results["raw_rl_path"] = raw_rl_path
    results["raw_rl_metrics"] = execute_route(
        env, IoTNetworkSimulator(seed=seed + 20000), raw_rl_path
    )
    results["evaluated_rl_path"] = evaluated_rl_path
    results["rl_path"] = evaluated_rl_path
    results["used_aco_fallback"] = used_aco_fallback
    return results


def main(num_episodes=150):
    """Run one episode-count experiment; rerun with 250 or 400 separately."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    seeds = [42, 123, 256, 512, 1024]
    print(f"Running simulation for {len(seeds)} seeds, {num_episodes} episodes each...")

    seed_outputs = []
    for index, seed in enumerate(seeds, start=1):
        result = run_single_seed(seed, num_episodes=num_episodes)
        seed_outputs.append(result)
        fallback_note = " (ACO safety fallback used)" if result["used_aco_fallback"] else ""
        print(f"Seed {index}/{len(seeds)}: raw RL path {result['raw_rl_path']}")
        print(f"  Evaluated hybrid path {result['evaluated_rl_path']}{fallback_note}")

    evaluator = RLAROEvaluator(seeds=seeds)
    evaluation = evaluator.evaluate(seed_outputs)
    evaluator.print_report(evaluation, rl_path=seed_outputs[0]["rl_path"])

    print("\n--- Raw RL-ARO policy metrics (before ACO fallback) ---")
    for metric, label in (
        ("delivery_time_min", "Delivery time (min)"),
        ("fleet_energy_wh", "Fleet energy (Wh)"),
        ("pdr_percent", "End-to-end PDR (%)"),
        ("latency_ms", "Route communication latency (ms)"),
    ):
        values = np.asarray([r["raw_rl_metrics"][metric] for r in seed_outputs], dtype=float)
        print(f"{label:32s}: {values.mean():.2f} ± {values.std(ddof=1):.2f}")
    print("Fallback rate: " + f"{np.mean([r['used_aco_fallback'] for r in seed_outputs])*100:.1f}%")

    dashboard_path = evaluator.generate_dashboard(evaluation, seed_outputs)
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
