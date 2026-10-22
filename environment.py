
import numpy as np
import networkx as nx


class DroneLogisticsEnvironment:
    def __init__(self, num_nodes=10, area_size_m=1000.0, seed=42):
        if num_nodes != 10:
            raise ValueError(
                "This environment has a fixed 10-node topology; "
                "num_nodes must be 10"
            )
        if not np.isfinite(area_size_m) or area_size_m <= 0:
            raise ValueError("area_size_m must be a finite positive number")

        self.num_nodes = num_nodes
        self.area_size_m = float(area_size_m)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Physical parameters
        self.cruise_speed_mps = 16.5
        self.max_speed_mps = 22.0
        self.battery_capacity_wh = 120.0
        self.base_power_w = 180.0
        self.payload_kg_default = 1.5
        self.t_max_s = 1800.0
        self.e_max_wh = 65.0

        self.graph = nx.Graph()
        self.node_positions = {}
        self.charging_hubs = {3, 6}

        self._build_airspace_topology()
        self.update_edge_costs()

    # ---------------------------------------------------------------
    # Topology
    # ---------------------------------------------------------------

    def _build_airspace_topology(self):
        """Build the fixed 10-node healthcare delivery network."""
        coords = [
            (80, 120),
            (260, 310),
            (220, 720),
            (480, 180),
            (510, 520),
            (460, 860),
            (740, 290),
            (780, 680),
            (690, 910),
            (940, 880),
        ]

        for i, (x, y) in enumerate(coords):
            self.node_positions[i] = np.array(
                [x, y], dtype=np.float64
            )
            self.graph.add_node(
                i,
                pos=(x, y),
                is_hub=(i in self.charging_hubs),
            )

        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (1, 4),
            (2, 4), (2, 5),
            (3, 4), (3, 6),
            (4, 5), (4, 6), (4, 7),
            (5, 7), (5, 8),
            (6, 7), (6, 9),
            (7, 8), (7, 9),
            (8, 9),
        ]

        for u, v in edges:
            distance = float(
                np.linalg.norm(
                    self.node_positions[u] - self.node_positions[v]
                )
            )
            congestion = float(self.rng.uniform(0.12, 0.55))
            wind = float(self.rng.uniform(0.88, 1.18))

            self.graph.add_edge(
                u,
                v,
                distance=distance,
                base_congestion=congestion,
                congestion=congestion,
                wind=wind,
                base_wind=wind,
                cost=0.0,
                time_min=0.0,
                energy_wh=0.0,
            )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    def _validate_endpoints(self, source, target):
        if (
            not isinstance(source, (int, np.integer))
            or not isinstance(target, (int, np.integer))
            or source not in self.graph
            or target not in self.graph
        ):
            raise ValueError(
                "source and target must be valid integer node IDs"
            )

    @staticmethod
    def _validate_nonnegative_finite(value, name):
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be finite and nonnegative"
            )

    # ---------------------------------------------------------------
    # Route cost and hop physics
    # ---------------------------------------------------------------

    def compute_optimal_route_cost(
        self, distance_m, speed_mps, energy_wh, lam=0.5
    ):
        """
        Combined normalized time-energy cost for one edge:

        D_ro = (1 - lambda) * (time / T_max)
             + lambda * (energy / E_max)
        """
        self._validate_nonnegative_finite(distance_m, "distance_m")
        self._validate_nonnegative_finite(energy_wh, "energy_wh")

        if not np.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("speed_mps must be finite and positive")

        if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be between 0 and 1")

        travel_time_s = distance_m / speed_mps
        time_norm = np.clip(
            travel_time_s / self.t_max_s, 0.0, 1.0
        )
        energy_norm = np.clip(
            energy_wh / self.e_max_wh, 0.0, 1.0
        )

        cost = (1.0 - lam) * time_norm + lam * energy_norm
        return float(np.clip(cost, 0.0, 1.0))

    def evaluate_hop_physics(
        self, u, v, payload_kg=None, dynamic_perturbation=0.0
    ):
        """Calculate deterministic physical metrics for one hop."""
        if not self.graph.has_edge(u, v):
            raise ValueError(
                f"No edge exists between nodes {u} and {v}"
            )

        if payload_kg is None:
            payload_kg = self.payload_kg_default

        if not np.isfinite(payload_kg) or payload_kg < 0:
            raise ValueError(
                "payload_kg must be finite and nonnegative"
            )

        if not np.isfinite(dynamic_perturbation):
            raise ValueError(
                "dynamic_perturbation must be finite"
            )

        edge = self.graph[u][v]
        distance = float(edge["distance"])

        congestion = float(
            np.clip(
                edge["congestion"] + dynamic_perturbation,
                0.05,
                0.95,
            )
        )
        wind = float(edge["wind"])

        # Congestion and wind affect effective flight speed.
        speed = (
            self.cruise_speed_mps
            * (1.0 - 0.28 * congestion)
            / max(0.75, wind)
        )
        speed = float(
            np.clip(speed, 8.0, self.max_speed_mps)
        )

        flight_time_s = distance / speed
        flight_time_min = flight_time_s / 60.0

        payload_factor = 1.0 + 0.22 * payload_kg
        power_w = (
            self.base_power_w
            * payload_factor
            * wind
            * (1.0 + 0.35 * congestion)
        )
        energy_wh = power_w * flight_time_s / 3600.0

        ideal_energy = (
            self.base_power_w
            * (distance / self.cruise_speed_mps)
            / 3600.0
        )
        energy_saved_wh = max(
            0.0, ideal_energy * 1.15 - energy_wh
        )

        d_ro = self.compute_optimal_route_cost(
            distance, speed, energy_wh
        )

        return {
            "distance_m": float(distance),
            "flight_time_min": float(flight_time_min),
            "flight_time_s": float(flight_time_s),
            "energy_wh": float(energy_wh),
            "energy_saved_wh": float(energy_saved_wh),
            "congestion": float(congestion),
            "speed_mps": float(speed),
            "d_ro": float(d_ro),
            "wind": float(wind),
        }

    def update_edge_costs(self, dynamic_perturbation=0.0):
        """
        Refresh edge metrics from the current environment.

        Edge costs are nonnegative, so they are valid weights for
        NetworkX shortest-path algorithms.
        """
        if not np.isfinite(dynamic_perturbation):
            raise ValueError(
                "dynamic_perturbation must be finite"
            )

        for u, v in self.graph.edges():
            phys = self.evaluate_hop_physics(
                u, v,
                dynamic_perturbation=dynamic_perturbation,
            )

            edge = self.graph[u][v]
            edge["cost"] = phys["d_ro"]
            edge["time_min"] = phys["flight_time_min"]
            edge["energy_wh"] = phys["energy_wh"]
            edge["congestion"] = phys["congestion"]

    # ---------------------------------------------------------------
    # Environmental changes
    # ---------------------------------------------------------------

    def apply_environmental_event(self, intensity=0.15):
        """Apply a random congestion and wind change to every edge."""
        if not np.isfinite(intensity) or intensity < 0:
            raise ValueError(
                "intensity must be finite and nonnegative"
            )

        for u, v in self.graph.edges():
            edge = self.graph[u][v]

            edge["congestion"] = float(
                np.clip(
                    edge["base_congestion"]
                    + self.rng.uniform(-intensity, intensity),
                    0.05,
                    0.95,
                )
            )

            edge["wind"] = float(
                edge["base_wind"]
                * self.rng.uniform(0.92, 1.10)
            )

        self.update_edge_costs()

    # ---------------------------------------------------------------
    # Shortest-path baseline
    # ---------------------------------------------------------------

    def get_dijkstra_route(self, source=0, target=9):
        """Find a route minimizing the sum of current edge costs."""
        self._validate_endpoints(source, target)
        self.update_edge_costs()

        try:
            return nx.shortest_path(
                self.graph,
                source=source,
                target=target,
                weight="cost",
                method="dijkstra",
            )
        except nx.NetworkXNoPath as exc:
            raise nx.NetworkXNoPath(
                f"No route exists from {source} to {target}"
            ) from exc

    # ---------------------------------------------------------------
    # Ant Colony Optimization baseline
    # ---------------------------------------------------------------

    def get_aco_route(
        self,
        source=0,
        target=9,
        n_ants=20,
        n_iterations=40,
        alpha=1.0,
        beta=2.5,
        rho=0.15,
    ):
        """
        Ant Colony Optimization using edge costs and pheromone.

        If no valid ACO route is found, return the Dijkstra route.
        The returned path always uses existing graph edges.
        """
        self._validate_endpoints(source, target)

        if (
            not isinstance(n_ants, (int, np.integer))
            or not isinstance(n_iterations, (int, np.integer))
            or n_ants <= 0
            or n_iterations <= 0
        ):
            raise ValueError(
                "n_ants and n_iterations must be positive integers"
            )

        if (
            not np.isfinite(alpha)
            or not np.isfinite(beta)
            or alpha < 0
            or beta < 0
            or not np.isfinite(rho)
            or not 0 <= rho < 1
        ):
            raise ValueError(
                "alpha and beta must be nonnegative, "
                "and rho must be in [0, 1)"
            )

        self.update_edge_costs()

        if source == target:
            return [source]

        pheromone = {
            (min(u, v), max(u, v)): 1.0
            for u, v in self.graph.edges()
        }

        best_path = None
        best_cost = float("inf")
        epsilon = 1e-12

        for _ in range(n_iterations):
            successful_paths = []

            for _ in range(n_ants):
                path = [source]
                visited = {source}
                current = source

                while current != target:
                    neighbors = [
                        node
                        for node in self.graph.neighbors(current)
                        if node not in visited
                    ]

                    if not neighbors:
                        break

                    log_weights = []

                    for neighbor in neighbors:
                        key = (
                            min(current, neighbor),
                            max(current, neighbor),
                        )
                        tau = max(pheromone[key], epsilon)
                        edge_cost = max(
                            self.graph[current][neighbor]["cost"],
                            epsilon,
                        )

                        # Log space prevents numerical overflow when
                        # combining pheromone and inverse edge costs.
                        log_weight = (
                            alpha * np.log(tau)
                            - beta * np.log(edge_cost)
                        )
                        log_weights.append(log_weight)

                    log_weights = np.asarray(
                        log_weights, dtype=np.float64
                    )
                    log_weights -= np.max(log_weights)

                    weights = np.exp(log_weights)
                    weight_sum = weights.sum()

                    if (
                        not np.all(np.isfinite(weights))
                        or not np.isfinite(weight_sum)
                        or weight_sum <= 0
                    ):
                        probabilities = np.full(
                            len(neighbors),
                            1.0 / len(neighbors),
                        )
                    else:
                        probabilities = weights / weight_sum

                    next_node = int(
                        self.rng.choice(
                            neighbors, p=probabilities
                        )
                    )

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                if path[-1] != target:
                    continue

                # Evaluate the complete route, not just the partial
                # cost accumulated during construction.
                total_cost = sum(
                    self.graph[u][v]["cost"]
                    for u, v in zip(path, path[1:])
                )

                if not np.isfinite(total_cost):
                    continue

                successful_paths.append((path, total_cost))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = list(path)

            # Evaporate pheromone.
            for key in pheromone:
                pheromone[key] *= 1.0 - rho

            # Deposit pheromone only for complete successful paths.
            for path, cost in successful_paths:
                deposit = 1.0 / max(cost, epsilon)

                for u, v in zip(path, path[1:]):
                    key = (min(u, v), max(u, v))
                    pheromone[key] += deposit

        if best_path is None:
            return self.get_dijkstra_route(source, target)

        return best_path

    # ---------------------------------------------------------------
    # TSP-D-inspired greedy baseline
    # ---------------------------------------------------------------

    def get_tspd_route(self, source=0, target=9):
        """Greedy cost-plus-distance heuristic."""
        self._validate_endpoints(source, target)
        self.update_edge_costs()

        if source == target:
            return [source]

        path = [source]
        visited = {source}
        current = source

        while current != target:
            neighbors = [
                node
                for node in self.graph.neighbors(current)
                if node not in visited
            ]

            if not neighbors:
                # Use a valid shortest-path continuation if the greedy
                # choices reach a dead end.
                subpath = nx.shortest_path(
                    self.graph,
                    source=current,
                    target=target,
                    weight="cost",
                    method="dijkstra",
                )
                path.extend(subpath[1:])
                break

            def score(node):
                edge_cost = self.graph[current][node]["cost"]
                distance_to_goal = np.linalg.norm(
                    self.node_positions[node]
                    - self.node_positions[target]
                )
                return edge_cost + 0.0009 * distance_to_goal

            next_node = min(neighbors, key=score)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    # ---------------------------------------------------------------
    # Congestion-weighted baseline
    # ---------------------------------------------------------------

    def get_dbscan_route(self, source=0, target=9):
        """
        Congestion-weighted shortest path.

        Note: this is a congestion-weighted routing heuristic, not
        a full DBSCAN clustering implementation.
        """
        self._validate_endpoints(source, target)
        self.update_edge_costs()

        if source == target:
            return [source]

        try:
            return nx.shortest_path(
                self.graph,
                source=source,
                target=target,
                weight=lambda u, v, data: (
                    data["cost"]
                    * (1.0 + 1.8 * data["congestion"])
                ),
                method="dijkstra",
            )
        except nx.NetworkXNoPath as exc:
            raise nx.NetworkXNoPath(
                f"No route exists from {source} to {target}"
            ) from exc

    # ---------------------------------------------------------------
    # Agent observations
    # ---------------------------------------------------------------

    def get_neighbors_info(self, node):
        """
        Return (neighbor_id, energy_norm, congestion, qos).

        Tuple structure is preserved for compatibility with agent.py.
        """
        if node not in self.graph:
            raise ValueError("node must be a valid node ID")

        info = []

        for neighbor in sorted(self.graph.neighbors(node)):
            phys = self.evaluate_hop_physics(node, neighbor)

            energy_norm = float(
                np.clip(
                    phys["energy_wh"] / self.battery_capacity_wh,
                    0.0,
                    1.0,
                )
            )
            qos = float(1.0 - 0.6 * phys["congestion"])

            info.append(
                (
                    int(neighbor),
                    energy_norm,
                    phys["congestion"],
                    qos,
                )
            )

        return info
