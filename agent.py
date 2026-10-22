import numpy as np
from collections import deque


class MultiHeadAttention:
    """Multi-head attention in pure NumPy."""

    def __init__(self, d_model=16, num_heads=4, seed=42):
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model and num_heads must be positive, and d_model divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.default_rng(seed)

        self.W_Q = rng.normal(0, 0.1, (num_heads, d_model, self.d_k))
        self.W_K = rng.normal(0, 0.1, (num_heads, d_model, self.d_k))
        self.W_V = rng.normal(0, 0.1, (num_heads, d_model, self.d_k))
        self.W_O = rng.normal(0, 0.1, (d_model, d_model))

    def forward(self, states):
        """
        states: (N, d_model)  – local state first, then neighbors
        Returns context vector of shape (d_model,)
        """
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != self.d_model:
            raise ValueError(f"states must have shape (N, {self.d_model})")

        head_outputs = []

        for h in range(self.num_heads):
            Q = states @ self.W_Q[h]          # (N, d_k)
            K = states @ self.W_K[h]
            V = states @ self.W_V[h]

            scores = (Q @ K.T) / np.sqrt(self.d_k)
            scores -= np.max(scores, axis=-1, keepdims=True)  # numerical stability

            weights = np.exp(scores)
            weights /= np.sum(weights, axis=-1, keepdims=True) + 1e-8

            # Use the local agent's attended representation
            head_out = weights[0] @ V
            head_outputs.append(head_out)

        concat = np.concatenate(head_outputs)          # (d_model,)
        context = concat @ self.W_O
        return context


class DQN:
    """Deep Q-Network: input → 128 → 64 → 32 → output (full backprop)."""

    def __init__(self, input_dim, output_dim, seed=42):
        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(0, 0.1, (input_dim, 128))
        self.b1 = np.zeros(128)
        self.w2 = rng.normal(0, 0.1, (128, 64))
        self.b2 = np.zeros(64)
        self.w3 = rng.normal(0, 0.1, (64, 32))
        self.b3 = np.zeros(32)
        self.w4 = rng.normal(0, 0.1, (32, output_dim))
        self.b4 = np.zeros(output_dim)

    def forward(self, x):
        """Forward pass that also returns intermediates for backprop."""
        x = np.asarray(x, dtype=np.float64)

        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)              # ReLU

        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)

        z3 = h2 @ self.w3 + self.b3
        h3 = np.maximum(0, z3)

        q = h3 @ self.w4 + self.b4
        return q, (x, z1, h1, z2, h2, z3, h3)

    def copy_weights_from(self, other):
        for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
            setattr(self, name, getattr(other, name).copy())


class RLAROAgent:
    def __init__(
        self,
        num_nodes=10,
        state_dim=16,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=500,
        max_grad_norm=5.0,
        seed=42,
    ):
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if state_dim <= 0 or state_dim % 4 != 0:
            raise ValueError("state_dim must be positive and divisible by 4")
        if batch_size <= 0 or buffer_capacity <= 0 or target_update_freq <= 0:
            raise ValueError("batch_size, buffer_capacity, and target_update_freq must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm
        self.train_step = 0
        self.rng = np.random.default_rng(seed)

        self.attention = MultiHeadAttention(d_model=state_dim, num_heads=4, seed=seed)
        self.q_network = DQN(input_dim=state_dim, output_dim=num_nodes, seed=seed)
        self.target_network = DQN(input_dim=state_dim, output_dim=num_nodes, seed=seed + 1)
        self.target_network.copy_weights_from(self.q_network)

        self.replay_buffer = deque(maxlen=buffer_capacity)

    def _build_state_vector(self, node_id, energy_norm, congestion, qos):
        return np.array([
            node_id / self.num_nodes,
            energy_norm,
            congestion,
            qos
        ], dtype=np.float64)

    def encode_state(self, local_node, local_features, neighbor_list):
        """
        local_features: (energy_norm, congestion, qos)
        neighbor_list: list of tuples (neighbor_node_id, energy_norm, congestion, qos)
        """
        if not 0 <= local_node < self.num_nodes:
            raise ValueError("local_node is out of range")
        if len(local_features) != 3:
            raise ValueError("local_features must contain (energy_norm, congestion, qos)")

        def pad(vec):
            if len(vec) < self.state_dim:
                return np.pad(vec, (0, self.state_dim - len(vec)))
            return vec[:self.state_dim]

        states = [pad(self._build_state_vector(local_node, *local_features))]

        for neigh in neighbor_list:
            # neigh = (node_id, energy, congestion, qos)
            if len(neigh) != 4 or not 0 <= neigh[0] < self.num_nodes:
                raise ValueError("each neighbor must be (valid_node_id, energy, congestion, qos)")
            states.append(pad(self._build_state_vector(*neigh)))

        return self.attention.forward(np.stack(states))

    def compute_reward(
        self,
        energy_saved_norm,
        energy_consumed_norm,
        data_loss_rate,
        route_cost_dro=0.0,
        qos_index=0.9,
        auth_delay_ms=20.0,
        alpha=0.55,
        beta=0.45,
    ):
        """Route-focused reward with bounded, lower-weight link-quality terms.

        The dominant term is negative additive route cost, aligning reward
        accumulation with the environment's time/energy edge cost. Smaller
        terms account for energy and communication without rewarding extra
        hops with a large positive constant.
        """
        values = {
            "energy_saved_norm": energy_saved_norm,
            "energy_consumed_norm": energy_consumed_norm,
            "data_loss_rate": data_loss_rate,
            "route_cost_dro": route_cost_dro,
            "qos_index": qos_index,
            "auth_delay_ms": auth_delay_ms,
            "alpha": alpha,
            "beta": beta,
        }
        values = {name: float(value) for name, value in values.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("reward inputs must be finite")
        if not 0.0 <= values["data_loss_rate"] <= 1.0:
            raise ValueError("data_loss_rate must be in [0, 1]")
        if values["auth_delay_ms"] < 0:
            raise ValueError("auth_delay_ms must be nonnegative")

        energy_saved = float(np.clip(values["energy_saved_norm"], 0.0, 1.0))
        energy_used = float(np.clip(values["energy_consumed_norm"], 0.0, 1.0))
        route_cost = float(np.clip(values["route_cost_dro"], 0.0, 1.0))
        qos = float(np.clip(values["qos_index"], 0.0, 1.0))
        loss = values["data_loss_rate"]
        auth_norm = float(np.clip(values["auth_delay_ms"] / 50.0, 0.0, 1.0))

        # Route cost dominates; other factors are deliberately lower-weight.
        reward = (
            -2.0 * route_cost
            + 0.10 * values["alpha"] * (energy_saved - energy_used)
            - 0.10 * values["beta"] * loss
            + 0.05 * (qos - 0.80)
            - 0.02 * auth_norm
        )
        return float(reward)

    def select_action(self, state_vec, valid_neighbors, training=True):
        valid_neighbors = list(valid_neighbors)
        if not valid_neighbors:
            raise ValueError("valid_neighbors cannot be empty")
        if any(not isinstance(n, (int, np.integer)) or not 0 <= n < self.num_nodes
               for n in valid_neighbors):
            raise ValueError("valid_neighbors contains an invalid action index")

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_neighbors))

        q_values = self.q_network.forward(state_vec)[0]
        best_idx = int(np.argmax(q_values[valid_neighbors]))
        return int(valid_neighbors[best_idx])

    def store_transition(self, state, action, reward, next_state, done, next_valid_actions=None):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_nodes:
            raise ValueError("action must be an integer in [0, num_nodes)")
        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError(f"state and next_state must have shape ({self.state_dim},)")
        if next_valid_actions is None:
            next_valid_actions = list(range(self.num_nodes))
        else:
            next_valid_actions = list(next_valid_actions)
        if any(not isinstance(a, (int, np.integer)) or not 0 <= a < self.num_nodes
               for a in next_valid_actions):
            raise ValueError("next_valid_actions contains an invalid action index")
        if not done and not next_valid_actions:
            raise ValueError("nonterminal transitions must have at least one valid next action")

        self.replay_buffer.append((
            np.asarray(state, dtype=np.float64).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float64).copy(),
            bool(done),
            tuple(int(a) for a in next_valid_actions),
        ))

    def _clip_grads(self, grads):
        """Simple gradient clipping."""
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-8)
            return [g * scale for g in grads]
        return grads

    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        indices = self.rng.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        batch = [self.replay_buffer[int(i)] for i in indices]

        states      = np.stack([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=bool)
        next_valid_actions = [b[5] for b in batch]

        # Double-check target bootstrapping only considers legal next-hop actions.
        next_q = np.stack([self.target_network.forward(s)[0] for s in next_states])
        targets = rewards.copy()
        for i in range(self.batch_size):
            if not dones[i] and next_valid_actions[i]:
                legal_q = next_q[i, list(next_valid_actions[i])]
                targets[i] += self.gamma * np.max(legal_q)

        total_loss = 0.0

        # Accumulate gradients
        gw1 = np.zeros_like(self.q_network.w1)
        gb1 = np.zeros_like(self.q_network.b1)
        gw2 = np.zeros_like(self.q_network.w2)
        gb2 = np.zeros_like(self.q_network.b2)
        gw3 = np.zeros_like(self.q_network.w3)
        gb3 = np.zeros_like(self.q_network.b3)
        gw4 = np.zeros_like(self.q_network.w4)
        gb4 = np.zeros_like(self.q_network.b4)

        for i in range(self.batch_size):
            q_values, (x, z1, h1, z2, h2, z3, h3) = self.q_network.forward(states[i])

            # Error only on the taken action
            error = q_values[actions[i]] - targets[i]
            total_loss += error ** 2

            # Output gradient
            dq = np.zeros_like(q_values)
            dq[actions[i]] = 2.0 * error / self.batch_size

            # Layer 4
            gw4 += np.outer(h3, dq)
            gb4 += dq
            dh3 = dq @ self.q_network.w4.T
            dh3 *= (z3 > 0)               # ReLU backward

            # Layer 3
            gw3 += np.outer(h2, dh3)
            gb3 += dh3
            dh2 = dh3 @ self.q_network.w3.T
            dh2 *= (z2 > 0)

            # Layer 2
            gw2 += np.outer(h1, dh2)
            gb2 += dh2
            dh1 = dh2 @ self.q_network.w2.T
            dh1 *= (z1 > 0)

            # Layer 1
            gw1 += np.outer(x, dh1)
            gb1 += dh1

        # Clip and apply gradients
        grads = self._clip_grads([gw1, gb1, gw2, gb2, gw3, gb3, gw4, gb4])
        self.q_network.w1 -= self.lr * grads[0]
        self.q_network.b1 -= self.lr * grads[1]
        self.q_network.w2 -= self.lr * grads[2]
        self.q_network.b2 -= self.lr * grads[3]
        self.q_network.w3 -= self.lr * grads[4]
        self.q_network.b3 -= self.lr * grads[5]
        self.q_network.w4 -= self.lr * grads[6]
        self.q_network.b4 -= self.lr * grads[7]

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        return float(total_loss / self.batch_size)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
