
import numpy as np
from collections import deque


class MultiHeadAttention:
    """Multi-head attention implemented using NumPy."""

    def __init__(self, d_model=16, num_heads=4, seed=42):
        if (
            not isinstance(d_model, (int, np.integer))
            or not isinstance(num_heads, (int, np.integer))
            or d_model <= 0
            or num_heads <= 0
            or d_model % num_heads != 0
        ):
            raise ValueError(
                "d_model and num_heads must be positive integers, "
                "and d_model must be divisible by num_heads"
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_k = self.d_model // self.num_heads

        rng = np.random.default_rng(seed)

        self.W_Q = rng.normal(
            0, 0.1, (self.num_heads, self.d_model, self.d_k)
        )
        self.W_K = rng.normal(
            0, 0.1, (self.num_heads, self.d_model, self.d_k)
        )
        self.W_V = rng.normal(
            0, 0.1, (self.num_heads, self.d_model, self.d_k)
        )
        self.W_O = rng.normal(0, 0.1, (self.d_model, self.d_model))

    def forward(self, states):
        """Return the attended representation of the first state."""
        states = np.asarray(states, dtype=np.float64)

        if (
            states.ndim != 2
            or states.shape[0] == 0
            or states.shape[1] != self.d_model
        ):
            raise ValueError(
                f"states must have shape (N, {self.d_model}), with N >= 1"
            )

        if not np.all(np.isfinite(states)):
            raise ValueError("states must contain only finite values")

        head_outputs = []

        for h in range(self.num_heads):
            Q = states @ self.W_Q[h]
            K = states @ self.W_K[h]
            V = states @ self.W_V[h]

            scores = (Q @ K.T) / np.sqrt(self.d_k)
            scores -= np.max(scores, axis=-1, keepdims=True)

            weights = np.exp(scores)
            weights /= np.sum(weights, axis=-1, keepdims=True)

            head_outputs.append(weights[0] @ V)

        concat = np.concatenate(head_outputs)
        return concat @ self.W_O


class DQN:
    """DQN with architecture input -> 128 -> 64 -> 32 -> output."""

    def __init__(self, input_dim, output_dim, seed=42):
        if (
            not isinstance(input_dim, (int, np.integer))
            or not isinstance(output_dim, (int, np.integer))
            or input_dim <= 0
            or output_dim <= 0
        ):
            raise ValueError("input_dim and output_dim must be positive integers")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        rng = np.random.default_rng(seed)

        self.w1 = rng.normal(0, 0.1, (self.input_dim, 128))
        self.b1 = np.zeros(128)
        self.w2 = rng.normal(0, 0.1, (128, 64))
        self.b2 = np.zeros(64)
        self.w3 = rng.normal(0, 0.1, (64, 32))
        self.b3 = np.zeros(32)
        self.w4 = rng.normal(0, 0.1, (32, self.output_dim))
        self.b4 = np.zeros(self.output_dim)

    def forward(self, x):
        """Return Q-values and intermediates for backpropagation."""
        x = np.asarray(x, dtype=np.float64)

        if x.shape != (self.input_dim,):
            raise ValueError(f"x must have shape ({self.input_dim},)")

        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")

        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(0, z1)

        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, z2)

        z3 = h2 @ self.w3 + self.b3
        h3 = np.maximum(0, z3)

        q = h3 @ self.w4 + self.b4

        return q, (x, z1, h1, z2, h2, z3, h3)

    def copy_weights_from(self, other):
        if self.output_dim != other.output_dim or self.input_dim != other.input_dim:
            raise ValueError("Network architectures must match")

        for name in (
            "w1", "b1", "w2", "b2",
            "w3", "b3", "w4", "b4"
        ):
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
        if not isinstance(num_nodes, (int, np.integer)) or num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer")

        if (
            not isinstance(state_dim, (int, np.integer))
            or state_dim <= 0
            or state_dim % 4 != 0
        ):
            raise ValueError("state_dim must be positive and divisible by 4")

        if (
            not np.isfinite(learning_rate)
            or learning_rate <= 0
            or not np.isfinite(gamma)
            or not 0 <= gamma <= 1
        ):
            raise ValueError("learning_rate must be positive and gamma in [0, 1]")

        if (
            not np.isfinite(epsilon_start)
            or not np.isfinite(epsilon_end)
            or not 0 <= epsilon_end <= epsilon_start <= 1
            or not np.isfinite(epsilon_decay)
            or not 0 < epsilon_decay <= 1
        ):
            raise ValueError("Invalid epsilon parameters")

        if (
            not isinstance(batch_size, (int, np.integer))
            or not isinstance(buffer_capacity, (int, np.integer))
            or not isinstance(target_update_freq, (int, np.integer))
            or batch_size <= 0
            or buffer_capacity < batch_size
            or target_update_freq <= 0
        ):
            raise ValueError(
                "buffer_capacity must be >= batch_size; "
                "batch_size and target_update_freq must be positive"
            )

        if not np.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.num_nodes = int(num_nodes)
        self.state_dim = int(state_dim)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = float(max_grad_norm)

        self.train_step = 0
        self.rng = np.random.default_rng(seed)

        self.attention = MultiHeadAttention(
            d_model=self.state_dim, num_heads=4, seed=seed
        )
        self.q_network = DQN(
            input_dim=self.state_dim,
            output_dim=self.num_nodes,
            seed=seed + 1,
        )
        self.target_network = DQN(
            input_dim=self.state_dim,
            output_dim=self.num_nodes,
            seed=seed + 2,
        )
        self.target_network.copy_weights_from(self.q_network)

        self.replay_buffer = deque(maxlen=buffer_capacity)

    def _build_state_vector(self, node_id, energy_norm, congestion, qos):
        values = np.asarray(
            [energy_norm, congestion, qos], dtype=np.float64
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("Node features must be finite")

        return np.array(
            [
                node_id / self.num_nodes,
                values[0],
                values[1],
                values[2],
            ],
            dtype=np.float64,
        )

    def encode_state(self, local_node, local_features, neighbor_list):
        if (
            not isinstance(local_node, (int, np.integer))
            or not 0 <= local_node < self.num_nodes
        ):
            raise ValueError("local_node is out of range")

        if len(local_features) != 3:
            raise ValueError(
                "local_features must contain energy, congestion, and QoS"
            )

        def pad(vec):
            if len(vec) >= self.state_dim:
                return vec[:self.state_dim]
            return np.pad(vec, (0, self.state_dim - len(vec)))

        states = [
            pad(self._build_state_vector(local_node, *local_features))
        ]

        for neigh in neighbor_list:
            if (
                len(neigh) != 4
                or not isinstance(neigh[0], (int, np.integer))
                or not 0 <= neigh[0] < self.num_nodes
            ):
                raise ValueError(
                    "Each neighbor must be "
                    "(valid_node_id, energy, congestion, qos)"
                )

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

        try:
            values = {
                name: float(value) for name, value in values.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("Reward inputs must be numeric") from exc

        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Reward inputs must be finite")

        if not 0 <= values["data_loss_rate"] <= 1:
            raise ValueError("data_loss_rate must be in [0, 1]")

        if values["auth_delay_ms"] < 0:
            raise ValueError("auth_delay_ms must be nonnegative")

        if not 0 <= values["alpha"] <= 1 or not 0 <= values["beta"] <= 1:
            raise ValueError("alpha and beta must be in [0, 1]")

        energy_saved = np.clip(values["energy_saved_norm"], 0, 1)
        energy_used = np.clip(values["energy_consumed_norm"], 0, 1)
        route_cost = np.clip(values["route_cost_dro"], 0, 1)
        qos = np.clip(values["qos_index"], 0, 1)
        loss = values["data_loss_rate"]
        auth_norm = np.clip(values["auth_delay_ms"] / 50.0, 0, 1)

        reward = (
            -2.0 * route_cost
            + 0.10 * values["alpha"] * (energy_saved - energy_used)
            - 0.10 * values["beta"] * loss
            + 0.05 * (qos - 0.80)
            - 0.02 * auth_norm
        )

        return float(reward)

    def _validate_actions(self, actions, allow_empty=False):
        actions = list(actions)

        if not actions and not allow_empty:
            raise ValueError("At least one valid action is required")

        if any(
            not isinstance(a, (int, np.integer))
            or not 0 <= a < self.num_nodes
            for a in actions
        ):
            raise ValueError("Action indices must be valid node indices")

        return list(dict.fromkeys(int(a) for a in actions))

    def select_action(self, state_vec, valid_neighbors, training=True):
        valid_neighbors = self._validate_actions(valid_neighbors)

        state_vec = np.asarray(state_vec, dtype=np.float64)

        if state_vec.shape != (self.state_dim,):
            raise ValueError(f"state_vec must have shape ({self.state_dim},)")

        if not np.all(np.isfinite(state_vec)):
            raise ValueError("state_vec must contain only finite values")

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_neighbors))

        q_values = self.q_network.forward(state_vec)[0]
        return max(valid_neighbors, key=lambda a: q_values[a])

    def store_transition(
        self, state, action, reward, next_state, done,
        next_valid_actions=None
    ):
        if (
            not isinstance(action, (int, np.integer))
            or not 0 <= action < self.num_nodes
        ):
            raise ValueError("action must be a valid node index")

        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)

        if (
            state.shape != (self.state_dim,)
            or next_state.shape != (self.state_dim,)
        ):
            raise ValueError(
                f"state and next_state must have shape ({self.state_dim},)"
            )

        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(next_state)):
            raise ValueError("States must contain only finite values")

        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        if next_valid_actions is None:
            next_valid_actions = list(range(self.num_nodes))

        next_valid_actions = self._validate_actions(
            next_valid_actions, allow_empty=bool(done)
        )

        if not done and not next_valid_actions:
            raise ValueError("Nonterminal transitions require valid next actions")

        self.replay_buffer.append(
            (
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                bool(done),
                tuple(next_valid_actions),
            )
        )

    def _clip_grads(self, grads):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-8)
            return [g * scale for g in grads]

        return grads

    def _train_batch(self, states, actions, targets):
        """Perform one minibatch gradient-descent update."""
        batch_size = len(states)
        total_loss = 0.0

        grads = [
            np.zeros_like(getattr(self.q_network, name))
            for name in (
                "w1", "b1", "w2", "b2",
                "w3", "b3", "w4", "b4"
            )
        ]

        for i in range(batch_size):
            q_values, cache = self.q_network.forward(states[i])
            x, z1, h1, z2, h2, z3, h3 = cache

            error = q_values[actions[i]] - targets[i]
            total_loss += error ** 2

            dq = np.zeros_like(q_values)
            dq[actions[i]] = 2.0 * error / batch_size

            gw1, gb1, gw2, gb2, gw3, gb3, gw4, gb4 = grads

            gw4 += np.outer(h3, dq)
            gb4 += dq
            dh3 = (dq @ self.q_network.w4.T) * (z3 > 0)

            gw3 += np.outer(h2, dh3)
            gb3 += dh3
            dh2 = (dh3 @ self.q_network.w3.T) * (z2 > 0)

            gw2 += np.outer(h1, dh2)
            gb2 += dh2
            dh1 = (dh2 @ self.q_network.w2.T) * (z1 > 0)

            gw1 += np.outer(x, dh1)
            gb1 += dh1

        grads = self._clip_grads(grads)

        for name, grad in zip(
            ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"),
            grads,
        ):
            setattr(
                self.q_network,
                name,
                getattr(self.q_network, name) - self.lr * grad,
            )

        return float(total_loss / batch_size)

    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        indices = self.rng.choice(
            len(self.replay_buffer), size=self.batch_size, replace=False
        )
        batch = [self.replay_buffer[int(i)] for i in indices]

        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=int)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=bool)

        targets = rewards.copy()

        # Double DQN: online network selects; target network evaluates.
        for i, transition in enumerate(batch):
            if dones[i]:
                continue

            legal_actions = list(transition[5])
            if not legal_actions:
                raise ValueError(
                    "Nonterminal transition has no valid next actions"
                )

            online_next_q = self.q_network.forward(next_states[i])[0]
            best_next_action = max(
                legal_actions, key=lambda a: online_next_q[a]
            )

            target_next_q = self.target_network.forward(next_states[i])[0]
            targets[i] += self.gamma * target_next_q[best_next_action]

        loss = self._train_batch(states, actions, targets)

        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        return loss

    def decay_exploration(self):
        self.epsilon = max(
            self.epsilon_end, self.epsilon * self.epsilon_decay
        )

    def fit_action_targets(self, examples, updates=100):
        """
        Supervised fitting of selected action Q-values to teacher targets.

        examples: iterable of (state, action, target_q)
        Does not modify the RL replay buffer.
        """
        examples = list(examples)

        if not isinstance(updates, (int, np.integer)) or updates < 0:
            raise ValueError("updates must be a nonnegative integer")

        if not examples or updates == 0:
            return 0.0

        normalized = []

        for state, action, target_q in examples:
            state = np.asarray(state, dtype=np.float64)

            if state.shape != (self.state_dim,):
                raise ValueError(
                    f"state must have shape ({self.state_dim},)"
                )

            if not np.all(np.isfinite(state)):
                raise ValueError("state must contain only finite values")

            if (
                not isinstance(action, (int, np.integer))
                or not 0 <= action < self.num_nodes
            ):
                raise ValueError("action must be a valid node index")

            target_q = float(target_q)
            if not np.isfinite(target_q):
                raise ValueError("target_q must be finite")

            normalized.append((state.copy(), int(action), target_q))

        total_loss = 0.0

        for _ in range(int(updates)):
            indices = self.rng.choice(
                len(normalized),
                size=min(self.batch_size, len(normalized)),
                replace=len(normalized) < self.batch_size,
            )
            batch = [normalized[int(i)] for i in indices]

            states = np.stack([item[0] for item in batch])
            actions = np.array([item[1] for item in batch], dtype=int)
            targets = np.array([item[2] for item in batch], dtype=np.float64)

            total_loss += self._train_batch(states, actions, targets)

        self.target_network.copy_weights_from(self.q_network)

        return float(total_loss / updates)
