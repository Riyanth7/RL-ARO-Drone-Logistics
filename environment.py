import numpy as np
import networkx as nx


class DroneLogisticsEnvironment:
    def __init__(self, num_nodes=10, area_size_m=1000.0, seed=42):
        if num_nodes != 10:
            raise ValueError("This environment has a fixed 10-node topology; num_nodes must be 10")
        if area_size_m <= 0:
            raise ValueError("area_size_m must be positive")
        self.num_nodes = num_nodes
        self.area_size_m = area_size_m
        self.rng = np.random.default_rng(seed)

        # Physical parameters (closer to paper assumptions)
        self.cruise_speed_mps = 16.5          # m/s
        self.max_speed_mps = 22.0
        self.battery_capacity_wh = 120.0      # per drone
        self.base_power_w = 180.0             # hovering/cruising base
        self.payload_kg_default = 1.5
        self.t_max_s = 1800.0                 # max mission time
        self.e_max_wh = 65.0                  # reference energy for normalization

        self.graph = nx.Graph()
        self.node_positions = {}
        self.charging_hubs = {3, 6}           # nodes that act as charging hubs

        self._build_airspace_topology()
        self.update_edge_costs()

    def _build_airspace_topology(self):
        """10-node healthcare delivery network (matches paper scale)."""
        # Fixed realistic coordinates (in meters)
        coords = [
            (80, 120),   # 0: Central Medical Depot
            (260, 310),  # 1: Urban Clinic A
            (220, 720),  # 2: Urban Clinic B
            (480, 180),  # 3: Charging Hub Alpha
            (510, 520),  # 4: Emergency Trauma Center
            (460, 860),  # 5: Northern Vaccine Center
            (740, 290),  # 6: Charging Hub Beta
            (780, 680),  # 7: Blood Bank
            (690, 910),  # 8: Suburban Hospital
            (940, 880),  # 9: Remote Emergency Destination
        ]

        for i, (x, y) in enumerate(coords):
            self.node_positions[i] = np.array([x, y], dtype=np.float64)
            self.graph.add_node(i, pos=(x, y), is_hub=(i in self.charging_hubs))

        # Realistic connectivity
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (1, 4),
            (2, 4), (2, 5),
            (3, 4), (3, 6),
            (4, 5), (4, 6), (4, 7),
            (5, 7), (5, 8),
            (6, 7), (6, 9),
            (7, 8), (7, 9),
            (8, 9)
        ]

        for u, v in edges:
            dist = float(np.linalg.norm(self.node_positions[u] - self.node_positions[v]))
            congestion = float(self.rng.uniform(0.12, 0.55))
            wind = float(self.rng.uniform(0.88, 1.18))
            self.graph.add_edge(
                u, v,
                distance=dist,
                base_congestion=congestion,
                congestion=congestion,
                wind=wind,
                base_wind=wind,
                cost=0.0,
                time_min=0.0,
                energy_wh=0.0
            )

    def compute_optimal_route_cost(self, distance_m, speed_mps, energy_wh, lam=0.5):
        """
        Equation (2) from the paper:
        D_ro = (1 - λ) * (time / T_max) + λ * (energy / E_max)
        """
        if distance_m < 0 or energy_wh < 0 or speed_mps <= 0:
            raise ValueError("distance and energy must be nonnegative; speed must be positive")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be between 0 and 1")
        travel_time_s = distance_m / speed_mps
        time_norm = np.clip(travel_time_s / self.t_max_s, 0.0, 1.0)
        energy_norm = np.clip(energy_wh / self.e_max_wh, 0.0, 1.0)
        d_ro = (1.0 - lam) * time_norm + lam * energy_norm
        return float(np.clip(d_ro, 0.0, 1.0))

    def evaluate_hop_physics(self, u, v, payload_kg=None, dynamic_perturbation=0.0):
        """Realistic physics for one hop."""
        if not self.graph.has_edge(u, v):
            raise ValueError(f"No edge exists between nodes {u} and {v}")
        if payload_kg is not None and payload_kg < 0:
            raise ValueError("payload_kg must be nonnegative")
        if payload_kg is None:
            payload_kg = self.payload_kg_default

        edge = self.graph[u][v]
        dist = edge["distance"]
        congestion = np.clip(edge["congestion"] + dynamic_perturbation, 0.05, 0.95)
        wind = edge["wind"]

        # Effective speed decreases with congestion and adverse wind
        speed = self.cruise_speed_mps * (1.0 - 0.28 * congestion) / max(0.75, wind)
        speed = np.clip(speed, 8.0, self.max_speed_mps)

        flight_time_s = dist / speed
        flight_time_min = flight_time_s / 60.0

        # Energy model: base power + payload + wind + congestion penalty
        payload_factor = 1.0 + 0.22 * payload_kg
        power_w = self.base_power_w * payload_factor * wind * (1.0 + 0.35 * congestion)
        energy_wh = (power_w * flight_time_s) / 3600.0

        # Energy that could have been saved with ideal conditions
        ideal_energy = (self.base_power_w * (dist / self.cruise_speed_mps)) / 3600.0
        energy_saved_wh = max(0.0, ideal_energy * 1.15 - energy_wh)

        d_ro = self.compute_optimal_route_cost(dist, speed, energy_wh)

        return {
            "distance_m": float(dist),
            "flight_time_min": float(flight_time_min),
            "flight_time_s": float(flight_time_s),
            "energy_wh": float(energy_wh),
            "energy_saved_wh": float(energy_saved_wh),
            "congestion": float(congestion),
            "speed_mps": float(speed),
            "d_ro": float(d_ro),
            "wind": float(wind)
        }

    def update_edge_costs(self, dynamic_perturbation=0.0):
        """Update all edge costs using current physics (Eq. 2)."""
        for u, v in self.graph.edges():
            phys = self.evaluate_hop_physics(u, v, dynamic_perturbation=dynamic_perturbation)
            self.graph[u][v]["cost"] = phys["d_ro"]
            self.graph[u][v]["time_min"] = phys["flight_time_min"]
            self.graph[u][v]["energy_wh"] = phys["energy_wh"]
            self.graph[u][v]["congestion"] = phys["congestion"]

    def apply_environmental_event(self, intensity=0.15):
        """Simulate sudden weather / traffic change."""
        if intensity < 0:
            raise ValueError("intensity must be nonnegative")
        for u, v in self.graph.edges():
            delta = self.rng.uniform(-intensity, intensity)
            self.graph[u][v]["congestion"] = np.clip(
                self.graph[u][v]["base_congestion"] + delta, 0.05, 0.95
            )
            self.graph[u][v]["wind"] = self.graph[u][v]["base_wind"] * self.rng.uniform(0.92, 1.10)
        self.update_edge_costs()

    # ------------------------------------------------------------------
    # Baseline Algorithms (much stronger than original)
    # ------------------------------------------------------------------

    def _validate_endpoints(self, source, target):
        if source not in self.graph or target not in self.graph:
            raise ValueError("source and target must be valid node IDs in this graph")

    def get_dijkstra_route(self, source=0, target=9):
        """Classic shortest path on current costs."""
        self._validate_endpoints(source, target)
        self.update_edge_costs()
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight="cost")
        except nx.NetworkXNoPath as exc:
            raise nx.NetworkXNoPath(f"No route exists from {source} to {target}") from exc

    def get_aco_route(self, source=0, target=9, n_ants=20, n_iterations=40, alpha=1.0, beta=2.5, rho=0.15):
        """
        Proper Ant Colony Optimization baseline.
        """
        self._validate_endpoints(source, target)
        if n_ants <= 0 or n_iterations <= 0:
            raise ValueError("n_ants and n_iterations must be positive")
        if alpha < 0 or beta < 0 or not 0 <= rho < 1:
            raise ValueError("alpha and beta must be nonnegative; rho must be in [0, 1)")
        self.update_edge_costs()

        # Initialize pheromone
        pheromone = { (min(u,v), max(u,v)): 1.0 for u, v in self.graph.edges() }

        best_path = None
        best_cost = float("inf")

        for _ in range(n_iterations):
            all_paths = []

            for _ in range(n_ants):
                path = [source]
                visited = {source}
                current = source
                total_cost = 0.0

                while current != target and len(path) < self.num_nodes * 2:
                    neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
                    if not neighbors:
                        # force progress with Dijkstra fallback
                        try:
                            sub = nx.shortest_path(self.graph, current, target, weight="cost")
                            path.extend(sub[1:])
                        except nx.NetworkXNoPath:
                            pass
                        break

                    # Transition probabilities
                    probs = []
                    for n in neighbors:
                        key = (min(current, n), max(current, n))
                        tau = pheromone[key] ** alpha
                        eta = (1.0 / (self.graph[current][n]["cost"] + 1e-6)) ** beta
                        probs.append(tau * eta)

                    probs = np.array(probs)
                    probs /= probs.sum()

                    next_node = int(self.rng.choice(neighbors, p=probs))
                    total_cost += self.graph[current][next_node]["cost"]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                # Recompute complete route cost, including any Dijkstra fallback.
                if all(a in self.graph and b in self.graph[a] for a, b in zip(path, path[1:])):
                    total_cost = sum(self.graph[a][b]["cost"] for a, b in zip(path, path[1:]))
                else:
                    total_cost = float("inf")
                all_paths.append((path, total_cost))
                if total_cost < best_cost and path[-1] == target:
                    best_cost = total_cost
                    best_path = path

            # Pheromone update
            for key in pheromone:
                pheromone[key] *= (1.0 - rho)

            for path, cost in all_paths:
                if path[-1] != target:
                    continue
                deposit = 1.0 / (cost + 1e-6)
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    pheromone[key] += deposit

        if best_path is None:
            return self.get_dijkstra_route(source, target)
        return best_path

    def get_tspd_route(self, source=0, target=9):
        """TSP-D style greedy heuristic with look-ahead."""
        self._validate_endpoints(source, target)
        self.update_edge_costs()
        path = [source]
        visited = {source}
        current = source

        while current != target:
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                # fallback
                try:
                    sub = nx.shortest_path(self.graph, current, target, weight="cost")
                    path.extend(sub[1:])
                except nx.NetworkXNoPath as exc:
                    raise nx.NetworkXNoPath(
                        f"No route exists from {source} to {target}"
                    ) from exc
                break

            def score(n):
                edge_cost = self.graph[current][n]["cost"]
                dist_to_goal = np.linalg.norm(self.node_positions[n] - self.node_positions[target])
                return edge_cost + 0.0009 * dist_to_goal

            next_node = min(neighbors, key=score)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def get_dbscan_route(self, source=0, target=9):
        """Density-inspired route: prefer low-congestion corridors."""
        self._validate_endpoints(source, target)
        self.update_edge_costs()
        # Temporarily re-weight by congestion
        for u, v in self.graph.edges():
            self.graph[u][v]["temp_weight"] = (
                self.graph[u][v]["cost"] * (1.0 + 1.8 * self.graph[u][v]["congestion"])
            )
        try:
            path = nx.shortest_path(self.graph, source=source, target=target, weight="temp_weight")
        except nx.NetworkXNoPath:
            path = self.get_dijkstra_route(source, target)
        return path

    def get_neighbors_info(self, node):
        """Helper for the agent: returns list of (neighbor_id, energy_norm, congestion, qos)."""
        if node not in self.graph:
            raise ValueError("node must be a valid node ID in this graph")
        info = []
        for n in self.graph.neighbors(node):
            phys = self.evaluate_hop_physics(node, n)
            energy_norm = np.clip(phys["energy_wh"] / self.battery_capacity_wh, 0.0, 1.0)
            # Simple QoS proxy
            qos = 1.0 - 0.6 * phys["congestion"]
            info.append((n, energy_norm, phys["congestion"], qos))
        return info
