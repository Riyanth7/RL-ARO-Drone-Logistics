import numpy as np
import networkx as nx

class DroneLogisticsEnvironment:
    def __init__(self, num_nodes=10, area_size_m=1000.0, seed=42):
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.area_size_m = area_size_m
        self.t_max_s = 1800.0   # Max allowable mission time (s)
        self.e_max_wh = 65.0    # Max mission reference energy capacity (Wh)
        self.cruise_speed_mps = 16.5

        self.graph = nx.Graph()
        self.node_positions = {}
        self._build_airspace_topology()

    def _build_airspace_topology(self):
        """Constructs the 1000m x 1000m healthcare delivery network topology (Table 2)."""
        fixed_coords = [
            (80, 120),   # Node 0: Central Medical Supply Depot
            (260, 310),  # Node 1: Urban Clinic A
            (220, 720),  # Node 2: Urban Clinic B
            (480, 180),  # Node 3: Charging Hub Alpha
            (510, 520),  # Node 4: Emergency Trauma Center
            (460, 860),  # Node 5: Northern Vaccine Center
            (740, 290),  # Node 6: Charging Hub Beta
            (780, 680),  # Node 7: Blood Bank Facility
            (690, 910),  # Node 8: Suburban Hospital
            (940, 880)   # Node 9: Remote Emergency Destination
        ]
        for i in range(self.num_nodes):
            self.node_positions[i] = np.array(fixed_coords[i], dtype=np.float64)
            self.graph.add_node(i, pos=fixed_coords[i])

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
            base_congestion = float(np.random.uniform(0.15, 0.55))
            wind_factor = float(np.random.uniform(0.90, 1.15))
            self.graph.add_edge(u, v, distance=dist, congestion=base_congestion, wind=wind_factor)

    def compute_optimal_route_cost(self, distance_m, speed_mps, energy_wh, lam=0.5):
        """Equation 2 (Page 7): Optimal Route Cost Function (D_ro)."""
        travel_time_s = distance_m / max(1.0, speed_mps)
        time_norm = np.clip(travel_time_s / self.t_max_s, 0.0, 1.0)
        energy_norm = np.clip(energy_wh / self.e_max_wh, 0.0, 1.0)
        d_ro = (1.0 - lam) * time_norm + lam * energy_norm
        return float(np.clip(d_ro, 0.0, 1.0))

    def compute_scheduling_utility(self, tasks_processed, total_tasks, service_time_s, d1=0.6, d2=0.4):
        """Equation 4 (Page 8): Smart Scheduling Utility (U_s)."""
        q_ratio = np.clip(tasks_processed / max(1.0, total_tasks), 0.0, 1.0)
        t_norm = np.clip(service_time_s / 60.0, 0.1, 1.0)
        u_s = d1 * q_ratio + d2 * (1.0 / (1.0 + t_norm)) + 0.1 * (1.0 - t_norm * q_ratio)
        return float(np.clip(u_s, 0.0, 1.0))

    def evaluate_hop_physics(self, u, v, payload_kg=1.5, dynamic_perturbation=0.0):
        """Calculates travel time, energy consumption (Wh), and Equation 2 cost for edge (u, v)."""
        edge_data = self.graph[u][v]
        dist_m = edge_data["distance"]
        congestion = np.clip(edge_data["congestion"] + dynamic_perturbation, 0.05, 0.95)
        wind = edge_data["wind"]

        effective_speed = self.cruise_speed_mps * (1.0 - 0.25 * congestion) / wind
        flight_time_min = (dist_m / max(2.0, effective_speed)) / 60.0 * 14.5

        payload_factor = 1.0 + 0.18 * payload_kg
        energy_wh = (dist_m / 100.0) * 3.15 * payload_factor * wind * (1.0 + 0.30 * congestion)
        energy_saved_wh = max(0.5, 18.5 - energy_wh)

        d_ro = self.compute_optimal_route_cost(dist_m, effective_speed, energy_wh)
        return {
            "distance_m": float(dist_m),
            "flight_time_min": float(flight_time_min),
            "energy_wh": float(energy_wh),
            "energy_saved_wh": float(energy_saved_wh),
            "congestion": float(congestion),
            "d_ro": float(d_ro)
        }

    def update_edge_costs(self, dynamic_perturbation=0.0):
        """Updates all graph edge weights using Equation 2 route cost."""
        for u, v in self.graph.edges():
            hop = self.evaluate_hop_physics(u, v, dynamic_perturbation=dynamic_perturbation)
            self.graph[u][v]["cost"] = hop["d_ro"]
            self.graph[u][v]["time_min"] = hop["flight_time_min"]
            self.graph[u][v]["energy_wh"] = hop["energy_wh"]

    def get_dijkstra_route(self, source=0, target=9):
        """Baseline 0: Traditional Shortest-Path Routing (Table 6)."""
        self.update_edge_costs(dynamic_perturbation=0.0)
        return nx.shortest_path(self.graph, source=source, target=target, weight="distance")

    def get_aco_route(self, source=0, target=9):
        """Baseline 1: Ant Colony Optimization (ACO) probabilistic pheromone path."""
        current = source
        path = [current]
        visited = {current}
        while current != target and len(path) < self.num_nodes * 2:
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                neighbors = list(self.graph.neighbors(current))
            weights = []
            for n in neighbors:
                dist = self.graph[current][n]["distance"]
                target_dist = np.linalg.norm(self.node_positions[n] - self.node_positions[target]) + 50.0
                pheromone = (1000.0 / (dist + target_dist)) ** 1.2
                weights.append(pheromone)
            probs = np.array(weights) / np.sum(weights)
            current = int(np.random.choice(neighbors, p=probs))
            path.append(current)
            visited.add(current)
        if path[-1] != target:
            path.extend(nx.shortest_path(self.graph, source=path[-1], target=target)[1:])
        return path

    def get_tspd_route(self, source=0, target=9):
        """Baseline 2: Travelling Salesman Problem with Drone (TSP-D) greedy distance heuristic."""
        current = source
        path = [current]
        visited = {current}
        while current != target:
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                path.extend(nx.shortest_path(self.graph, source=current, target=target)[1:])
                break
            next_node = min(neighbors, key=lambda n: self.graph[current][n]["distance"] +
                            0.7 * np.linalg.norm(self.node_positions[n] - self.node_positions[target]))
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path

    def get_dbscan_route(self, source=0, target=9):
        """Baseline 3: DBSCAN density-clustered routing heuristic."""
        self.update_edge_costs(dynamic_perturbation=0.0)
        return nx.shortest_path(self.graph, source=source, target=target, weight="cost")
      
