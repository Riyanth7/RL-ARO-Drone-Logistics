
import numpy as np
from collections import deque


class MultiHeadAttention:
    """
    Lightweight multi-head attention encoder.

    This implementation preserves the original 16-dimensional state
    representation. Attention parameters are initialized reproducibly.
    """

    def __init__(self, d_model=16, num_heads=4, seed=42):
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        self.W_Q = rng.normal(
            0.0, scale, (num_heads, d_model, self.head_dim)
        )
        self.W_K = rng.normal(
            0.0, scale, (num_heads, d_model, self.head_dim)
        )
        self.W_V = rng.normal(
            0.0, scale, (num_heads, d_model, self.head_dim)
        )
        self.W_O = rng.normal(
            0.0, scale, (d_model, d_model)
        )

    def forward(self, features):
        """
        features: array of shape (number_of_nodes, d_model)
        returns: encoded representation of the first row.
        """
        features = np.asarray(features, dtype=np.float64)

        if features.ndim != 2 or features.shape[1] != self.d_model:
            raise ValueError(
                f"Expected features with shape (N, {self.d_model})"
            )

        if features.shape[0] == 0:
            return np.zeros(self.d_model, dtype=np.float64)

        if not np.all(np.isfinite(features)):
            raise ValueError("Attention input contains non-finite values")

        head_outputs = []

        for h in range(self.num_heads):
            Q = features @ self.W_Q[h]
            K = features @ self.W_K[h]
            V = features @ self.W_V[h]

            scores = Q @ K.T / np.sqrt(self.head_dim)

            # Numerically stable softmax.
            scores -= np.max(scores, axis=-1, keepdims=True)
            attention = np.exp(scores)
            denominator = attention.sum(axis=-1, keepdims=True)
            attention /= np.maximum(denominator, 1e-12)

            head_output = attention @ V
            head_outputs.append(head_output[0])

        combined = np.concatenate(head_outputs)
        encoded = combined @ self.W_O

        return np.asarray(encoded, dtype=np.float64)


class QNetwork:
    """NumPy-based multilayer perceptron for action-value estimation."""

    def __init__(
        self,
        input_dim=16,
        output_dim=10,
        hidden_dims=(128, 64, 32),
        seed=42,
    ):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Network dimensions must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = [input_dim, *hidden_dims, output_dim]

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for index in range(len(self.layer_dims) - 1):
            fan_in = self.layer_dims[index]
            fan_out = self.layer_dims[index + 1]

            # He initialization for hidden layers; Xavier for output.
            if index < len(self.layer_dims) - 2:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))

            self.weights.append(
                rng.normal(0.0, scale, (fan_in, fan_out))
            )
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    def forward(self, states):
        states = np.asarray(states, dtype=np.float64)

        if states.ndim == 1:
            states = states.reshape(1, -1)

        if states.ndim != 2 or states.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected states with shape (batch, {self.input_dim})"
            )

        activations = [states]
        pre_activations = []

        current = states

        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            z = current @ weight + bias
            pre_activations.append(z)

            if index < len(self.weights) - 1:
                current = np.maximum(z, 0.0)
            else:
                current = z

            activations.append(current)

        return current, (activations, pre_activations)

    def predict(self, states):
        return self.forward(states)[0]

    def train_batch(self, states, actions, targets, learning_rate,
                    max_grad_norm=5.0):
        """
        Update the network using mean squared TD error.

        Returns the pre-update batch loss.
        """
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)

        predictions, cache = self.forward(states)
        activations, pre_activations = cache

        batch_size = states.shape[0]
        if batch_size == 0:
            return 0.0

        row_indices = np.arange(batch_size)
        errors = predictions[row_indices, actions] - targets
        loss = float(np.mean(errors ** 2))

        # Derivative of mean squared error.
        delta = np.zeros_like(predictions)
        delta[row_indices, actions] = 2.0 * errors / batch_size

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        for layer in reversed(range(len(self.weights))):
            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0)

            if layer > 0:
                delta = delta @ self.weights[layer].T
                delta *= (pre_activations[layer - 1] > 0.0)

        # Clip the global gradient norm.
        squared_norm = sum(
            np.sum(gradient ** 2)
            for gradient in grad_weights + grad_biases
        )
        grad_norm = np.sqrt(squared_norm)

        if grad_norm > max_grad_norm:
            factor = max_grad_norm / (grad_norm + 1e-12)
            grad_weights = [g * factor for g in grad_weights]
            grad_biases = [g * factor for g in grad_biases]

        for layer in range(len(self.weights)):
            self.weights[layer] -= (
                learning_rate * grad_weights[layer]
            )
            self.biases[layer] -= (
                learning_rate * grad_biases[layer]
            )

        return loss

    def copy_from(self, other):
        """Copy parameters from another network."""
        if self.layer_dims != other.layer_dims:
            raise ValueError("Cannot copy networks with different shapes")

        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]


class RLAROAgent:
    """
    Reinforcement-learning agent for the 10-node drone-routing task.

    The public method names and primary argument formats are retained
    for compatibility with the existing training script.
    """

    def __init__(
        self,
        num_nodes=10,
        state_dim=16,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        replay_buffer_size=50000,
        batch_size=64,
        target_update_freq=500,
        seed=42,
        max_grad_norm=5.0,
    ):
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if state_dim != 16:
            raise ValueError(
                "This implementation uses a 16-dimensional state"
            )
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Invalid epsilon values")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if replay_buffer_size <= 0 or batch_size <= 0:
            raise ValueError("Buffer size and batch size must be positive")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.num_nodes = int(num_nodes)
        self.state_dim = int(state_dim)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)

        self.epsilon_start = float(epsilon_start)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = float(max_grad_norm)

        self.rng = np.random.default_rng(seed)

        self.attention = MultiHeadAttention(
            d_model=state_dim,
            num_heads=4,
            seed=seed,
        )

        self.q_network = QNetwork(
            input_dim=state_dim,
            output_dim=num_nodes,
            seed=seed + 1,
        )
        self.target_network = QNetwork(
            input_dim=state_dim,
            output_dim=num_nodes,
            seed=seed + 2,
        )
        self.target_network.copy_from(self.q_network)

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.update_count = 0
        self.last_loss = 0.0

    # ---------------------------------------------------------------
    # State encoding
    # ---------------------------------------------------------------

    @staticmethod
    def _build_state_vector(node_id, local_features, num_nodes):
        """
        Encode a node and its local features into four values.

        local_features: (energy_norm, congestion, qos)
        """
        if not 0 <= int(node_id) < num_nodes:
            raise ValueError("node_id is outside the valid node range")

        features = np.asarray(local_features, dtype=np.float64)

        if features.shape != (3,):
            raise ValueError(
                "local_features must contain energy, congestion, and QoS"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("Local features contain non-finite values")

        energy_norm, congestion, qos = features

        return np.array(
            [
                float(node_id) / max(num_nodes - 1, 1),
                np.clip(energy_norm, 0.0, 1.0),
                np.clip(congestion, 0.0, 1.0),
                np.clip(qos, 0.0, 1.0),
            ],
            dtype=np.float64,
        )

    def encode_state(
        self,
        node_id,
        local_features,
        neighbors_info,
        target_node=None,
        visited_nodes=None,
    ):
        """
        Encode the current node and its neighbors.

        Existing calls using three arguments remain valid.
        Optional target and visited information can be supplied by
        an updated training script.

        Each neighbor tuple:
            (neighbor_id, energy_norm, congestion, qos)
        """
        if not 0 <= int(node_id) < self.num_nodes:
            raise ValueError("Invalid current node")

        if target_node is not None and not (
            0 <= int(target_node) < self.num_nodes
        ):
            raise ValueError("Invalid target node")

        visited = set(visited_nodes or [])
        if any(
            not isinstance(n, (int, np.integer))
            or n < 0
            or n >= self.num_nodes
            for n in visited
        ):
            raise ValueError("visited_nodes contains an invalid node")

        rows = [
            self._build_state_vector(
                node_id, local_features, self.num_nodes
            )
        ]

        # Sort to make the representation independent of neighbor
        # iteration order. Use the same input construction for current
        # and next states.
        for item in sorted(neighbors_info, key=lambda x: int(x[0])):
            if len(item) != 4:
                raise ValueError(
                    "Each neighbor must contain four values"
                )

            neighbor_id, energy_norm, congestion, qos = item

            if not 0 <= int(neighbor_id) < self.num_nodes:
                raise ValueError("Invalid neighbor node ID")

            rows.append(
                self._build_state_vector(
                    neighbor_id,
                    (energy_norm, congestion, qos),
                    self.num_nodes,
                )
            )

        encoded = self.attention.forward(np.asarray(rows))

        # Add explicit destination and visited-node information when
        # supplied. These features replace otherwise unused padding.
        state = np.zeros(self.state_dim, dtype=np.float64)
        state[:4] = self._build_state_vector(
            node_id, local_features, self.num_nodes
        )

        state[4:8] = encoded[:4]

        if target_node is not None:
            state[8] = float(target_node) / max(self.num_nodes - 1, 1)
            state[9] = float(node_id == target_node)

        if visited_nodes is not None:
            for node in visited:
                state[10 + node] = 1.0

        # Preserve the expected fixed-size state.
        return state

    # ---------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------

    def select_action(self, state, valid_actions, training=True):
        """Epsilon-greedy action selection over valid neighbors."""
        valid_actions = list(dict.fromkeys(int(a) for a in valid_actions))

        if not valid_actions:
            raise ValueError("valid_actions cannot be empty")

        if any(a < 0 or a >= self.num_nodes for a in valid_actions):
            raise ValueError("valid_actions contains an invalid node")

        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape ({self.state_dim},)"
            )
        if not np.all(np.isfinite(state)):
            raise ValueError("State contains non-finite values")

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        q_values = self.q_network.predict(state)[0]
        valid_q_values = np.asarray(
            [q_values[a] for a in valid_actions]
        )

        # Deterministic tie-breaking for reproducible evaluation.
        best_index = int(np.argmax(valid_q_values))
        return valid_actions[best_index]

    # ---------------------------------------------------------------
    # Reward function
    # ---------------------------------------------------------------

    def compute_reward(
        self,
        energy_saved_norm,
        energy_consumed_norm,
        data_loss_rate,
        route_cost_dro,
        qos_index,
        auth_delay_ms,
        alpha=0.55,
        beta=0.45,
    ):
        """Compute the original weighted routing reward."""
        values = [
            energy_saved_norm,
            energy_consumed_norm,
            data_loss_rate,
            route_cost_dro,
            qos_index,
            auth_delay_ms,
            alpha,
            beta,
        ]

        if not np.all(np.isfinite(values)):
            raise ValueError("Reward inputs must all be finite")

        if not 0 <= data_loss_rate <= 1:
            raise ValueError("data_loss_rate must be in [0, 1]")
        if auth_delay_ms < 0:
            raise ValueError("auth_delay_ms must be nonnegative")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be nonnegative")

        energy_saved = np.clip(energy_saved_norm, 0.0, 1.0)
        energy_used = np.clip(energy_consumed_norm, 0.0, 1.0)
        route_cost = np.clip(route_cost_dro, 0.0, 1.0)
        qos = np.clip(qos_index, 0.0, 1.0)
        loss = np.clip(data_loss_rate, 0.0, 1.0)

        auth_norm = np.clip(auth_delay_ms / 100.0, 0.0, 1.0)

        reward = (
            -2.0 * route_cost
            + 0.10 * alpha * (energy_saved - energy_used)
            - 0.10 * beta * loss
            + 0.05 * (qos - 0.80)
            - 0.02 * auth_norm
        )

        return float(reward)

    # ---------------------------------------------------------------
    # Replay buffer
    # ---------------------------------------------------------------

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions,
    ):
        """Store a transition with independent state copies."""
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        next_state = np.asarray(
            next_state, dtype=np.float64
        ).reshape(-1)

        if state.shape != (self.state_dim,):
            raise ValueError("Invalid state shape")
        if next_state.shape != (self.state_dim,):
            raise ValueError("Invalid next_state shape")
        if not np.all(np.isfinite(state)) or not np.all(
            np.isfinite(next_state)
        ):
            raise ValueError("Transition contains non-finite state values")
        if not 0 <= int(action) < self.num_nodes:
            raise ValueError("Invalid action")
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")

        valid_next = tuple(
            dict.fromkeys(int(a) for a in next_valid_actions)
        )
        if any(a < 0 or a >= self.num_nodes for a in valid_next):
            raise ValueError("Invalid next action")

        self.replay_buffer.append(
            (
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                bool(done),
                valid_next,
            )
        )

    # ---------------------------------------------------------------
    # Double DQN update
    # ---------------------------------------------------------------

    def update_policy(self):
        """
        Perform one Double DQN replay update.

        The online network selects the next action; the target network
        evaluates it. Terminal transitions never bootstrap.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        indices = self.rng.choice(
            len(self.replay_buffer),
            size=self.batch_size,
            replace=False,
        )
        batch = [self.replay_buffer[i] for i in indices]

        states = np.stack([item[0] for item in batch])
        actions = np.asarray(
            [item[1] for item in batch], dtype=np.int64
        )
        rewards = np.asarray(
            [item[2] for item in batch], dtype=np.float64
        )
        next_states = np.stack([item[3] for item in batch])
        dones = np.asarray(
            [item[4] for item in batch], dtype=bool
        )
        next_valid_actions = [item[5] for item in batch]

        # Start with immediate rewards. This automatically prevents
        # bootstrapping for terminal transitions.
        targets = rewards.copy()

        nonterminal_indices = [
            i for i in range(len(batch))
            if not dones[i] and next_valid_actions[i]
        ]

        if nonterminal_indices:
            selected_next_states = next_states[nonterminal_indices]

            # Double DQN: online network chooses the action.
            online_next_q = self.q_network.predict(
                selected_next_states
            )

            # Target network evaluates the chosen action.
            target_next_q = self.target_network.predict(
                selected_next_states
            )

            for local_i, batch_i in enumerate(nonterminal_indices):
                valid_actions = next_valid_actions[batch_i]

                valid_online_q = online_next_q[local_i, list(valid_actions)]
                best_action = valid_actions[
                    int(np.argmax(valid_online_q))
                ]

                targets[batch_i] += (
                    self.gamma * target_next_q[local_i, best_action]
                )

        self.last_loss = self.q_network.train_batch(
            states=states,
            actions=actions,
            targets=targets,
            learning_rate=self.lr,
            max_grad_norm=self.max_grad_norm,
        )

        self.update_count += 1

        if self.update_count % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        return self.last_loss

    # ---------------------------------------------------------------
    # Exploration schedule
    # ---------------------------------------------------------------

    def decay_epsilon(self):
        """Reduce exploration after each training episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )
        return self.epsilon
