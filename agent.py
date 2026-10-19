import numpy as np

class RLAROAgent:
    def __init__(self, num_nodes=10, learning_rate=0.05, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.96, seed=42):
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-Table storing action values for (current_node, next_node, congestion_tier)
        self.q_table = np.zeros((num_nodes, num_nodes, 3), dtype=np.float64)

        # Algorithm 1 (Page 10): Multi-Head Attention Projection Weights (H = 2 heads)
        self.num_heads = 2
        self.d_k = 4
        self.W_Q = [np.random.uniform(0.1, 0.5, (self.d_k, 4)) for _ in range(self.num_heads)]
        self.W_K = [np.random.uniform(0.1, 0.5, (self.d_k, 4)) for _ in range(self.num_heads)]
        self.W_V = [np.random.uniform(0.1, 0.5, (self.d_k, 4)) for _ in range(self.num_heads)]

        # Experience Replay Buffer D (Algorithm 1, Line 1 & 16)
        self.replay_buffer = []
        self.buffer_capacity = 5000

    def compute_attention_context(self, local_state_vec, neighbor_state_vecs):
        """Algorithm 1 (Lines 6-11): Multi-Head Attention over neighbor UAV states."""
        head_outputs = []
        all_states = [local_state_vec] + neighbor_state_vecs
        for h in range(self.num_heads):
            q_i = np.dot(self.W_Q[h], local_state_vec)
            scores = []
            values = []
            for s_j in all_states:
                k_j = np.dot(self.W_K[h], s_j)
                v_j = np.dot(self.W_V[h], s_j)
                score = np.dot(q_i, k_j) / np.sqrt(self.d_k)
                scores.append(score)
                values.append(v_j)
            exp_scores = np.exp(scores - np.max(scores))
            alpha_ij = exp_scores / np.sum(exp_scores)
            h_i = np.sum([alpha_ij[idx] * values[idx] for idx in range(len(values))], axis=0)
            head_outputs.append(h_i)

        concat_context = np.concatenate(head_outputs)
        context_magnitude = float(np.mean(concat_context))
        if context_magnitude < 0.35:
            return 0  # Low congestion context
        elif context_magnitude < 0.70:
            return 1  # Medium congestion context
        return 2      # High congestion context

    def compute_reward(self, energy_saved_norm, energy_consumed_norm, data_loss_rate,
                       route_cost_dro, qos_index, auth_delay_ms, alpha=0.55, beta=0.45):
        """Equation 1 (Page 7): RL-ARO Reward Function with IoT & Security shaping."""
        base_reward = alpha * (energy_saved_norm - energy_consumed_norm) + beta * (1.0 - data_loss_rate)
        efficiency_bonus = 0.35 * (1.0 - route_cost_dro) + 0.20 * qos_index
        security_penalty = 0.05 * (auth_delay_ms / 50.0)
        return float(base_reward + efficiency_bonus - security_penalty)

    def select_action(self, current_node, valid_neighbors, congestion_tier, edge_heuristics=None, training=True):
        """Algorithm 1 (Line 13): Epsilon-Greedy action selection with heuristic guidance."""
        if not valid_neighbors:
            return current_node

        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_neighbors))

        best_node = valid_neighbors[0]
        best_val = -1e9
        for neighbor in valid_neighbors:
            q_val = self.q_table[current_node, neighbor, congestion_tier]
            heuristic_bonus = 0.0
            if edge_heuristics and neighbor in edge_heuristics:
                heuristic_bonus = edge_heuristics[neighbor]
            total_val = q_val + heuristic_bonus
            if total_val > best_val:
                best_val = total_val
                best_node = neighbor
        return int(best_node)

    def store_transition(self, state_tuple, action, reward, next_state_tuple, done):
        """Algorithm 1 (Line 16): Store transition in Experience Replay Buffer D."""
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state_tuple, action, reward, next_state_tuple, done))

    def update_policy(self, batch_size=32):
        """Algorithm 1 (Line 17): Temporal-Difference (TD) Q-Learning update from buffer."""
        if len(self.replay_buffer) == 0:
            return 0.0

        sample_size = min(len(self.replay_buffer), batch_size)
        indices = np.random.choice(len(self.replay_buffer), sample_size, replace=False)
        td_losses = []

        for idx in indices:
            (curr_node, c_tier), action, reward, (next_node, next_tier), done = self.replay_buffer[idx]
            current_q = self.q_table[curr_node, action, c_tier]
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_node, :, next_tier])
            td_error = target_q - current_q
            self.q_table[curr_node, action, c_tier] += self.lr * td_error
            td_losses.append(abs(td_error))

        return float(np.mean(td_losses))

    def decay_exploration(self):
        """Linearly/Exponentially decay exploration rate epsilon across episodes."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                             
