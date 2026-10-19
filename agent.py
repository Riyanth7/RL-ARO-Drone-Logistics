
import numpy as np
from collections import deque


class RLAROAgent:
    """Reinforcement-learning agent for adaptive drone-logistics routing."""

    def __init__(
        self,
        num_nodes=10,
        learning_rate=0.05,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.96,
        seed=42,
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError(
                "Require 0 <= epsilon_end <= epsilon_start <= 1."
            )
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")

        self.num_nodes = int(num_nodes)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        # A dedicated RNG avoids resetting NumPy's global random state.
        self.rng = np.random.default_rng(seed)

        # Q[current_node, action_node, congestion_tier]
        self.q_table = np.zeros(
            (self.num_nodes, self.num_nodes, 3),
            dtype=np.float64,
        )

        # Two attention heads, each projecting 4-D state vectors.
        self.num_heads = 2
        self.d_k = 4

        self.W_Q = [
            self.rng.uniform(0.1, 0.5, (self.d_k, 4))
            for _ in range(self.num_heads)
        ]
        self.W_K = [
            self.rng.uniform(0.1, 0.5, (self.d_k, 4))
            for _ in range(self.num_heads)
        ]
        self.W_V = [
            self.rng.uniform(0.1, 0.5, (self.d_k, 4))
            for _ in range(self.num_heads)
        ]

        # Bounded replay memory.
        self.buffer_capacity = 5000
        self.replay_buffer = deque(maxlen=self.buffer_capacity)

    def _validate_node(self, node):
        """Validate and normalize a node index."""
        if not isinstance(node, (int, np.integer)):
            raise TypeError("Node indices must be integers.")

        node = int(node)

        if not 0 <= node < self.num_nodes:
            raise ValueError(
                f"Node index {node} is outside "
                f"the valid range [0, {self.num_nodes - 1}]."
            )

        return node

    @staticmethod
    def _validate_tier(tier):
        """Validate a congestion tier in {0, 1, 2}."""
        if not isinstance(tier, (int, np.integer)):
            raise TypeError("Congestion tier must be an integer.")

        tier = int(tier)

        if tier not in (0, 1, 2):
            raise ValueError("Congestion tier must be 0, 1, or 2.")

        return tier

    @staticmethod
    def _state_vector(vector, name):
        """Ensure a state vector has four finite numerical features."""
        vector = np.asarray(vector, dtype=np.float64)

        if vector.shape != (4,):
            raise ValueError(
                f"{name} must have shape (4,), got {vector.shape}."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains non-finite values.")

        return vector

    def compute_attention_context(
        self, local_state_vec, neighbor_state_vecs
    ):
        """
        Compute two-head attention over the local node and its neighbors.

        Returns:
            0: low context magnitude
            1: medium context magnitude
            2: high context magnitude
        """
        local_state_vec = self._state_vector(
            local_state_vec, "local_state_vec"
        )

        neighbors = [
            self._state_vector(vec, "neighbor_state_vec")
            for vec in neighbor_state_vecs
        ]

        all_states = [local_state_vec] + neighbors
        head_outputs = []

        for h in range(self.num_heads):
            q_i = self.W_Q[h] @ local_state_vec

            keys = np.stack([
                self.W_K[h] @ state for state in all_states
            ])
            values = np.stack([
                self.W_V[h] @ state for state in all_states
            ])

            # FIX: Convert the scores to a NumPy array before
            # subtracting their maximum.
            scores = (keys @ q_i) / np.sqrt(self.d_k)
            scores = scores - np.max(scores)

            exp_scores = np.exp(scores)
            denominator = np.sum(exp_scores)

            if not np.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError(
                    "Attention normalization failed."
                )

            attention_weights = exp_scores / denominator
            head_output = np.sum(
                attention_weights[:, None] * values,
                axis=0,
            )

            head_outputs.append(head_output)

        concat_context = np.concatenate(head_outputs)
        context_magnitude = float(np.mean(concat_context))

        if context_magnitude < 0.35:
            return 0
        elif context_magnitude < 0.70:
            return 1

        return 2

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
        """
        Compute the RL-ARO reward using the supplied reward equation.

        Inputs must use consistent normalization and units.
        """
        values = np.asarray(
            [
                energy_saved_norm,
                energy_consumed_norm,
                data_loss_rate,
                route_cost_dro,
                qos_index,
                auth_delay_ms,
                alpha,
                beta,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("Reward inputs must all be finite.")

        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative.")

        base_reward = (
            alpha * (energy_saved_norm - energy_consumed_norm)
            + beta * (1.0 - data_loss_rate)
        )

        efficiency_bonus = (
            0.35 * (1.0 - route_cost_dro)
            + 0.20 * qos_index
        )

        security_penalty = 0.05 * (auth_delay_ms / 50.0)

        return float(
            base_reward + efficiency_bonus - security_penalty
        )

    def select_action(
        self,
        current_node,
        valid_neighbors,
        congestion_tier,
        edge_heuristics=None,
        training=True,
    ):
        """
        Select a neighboring node using epsilon-greedy Q-learning
        with optional heuristic guidance.
        """
        current_node = self._validate_node(current_node)
        congestion_tier = self._validate_tier(congestion_tier)

        if not isinstance(training, (bool, np.bool_)):
            raise TypeError("training must be a boolean.")

        neighbors = list(dict.fromkeys(
            self._validate_node(n) for n in valid_neighbors
        ))

        if not neighbors:
            return current_node

        if current_node in neighbors:
            raise ValueError(
                "A node cannot be its own valid neighbor."
            )

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(neighbors))

        best_node = neighbors[0]
        best_value = -np.inf

        for neighbor in neighbors:
            q_value = self.q_table[
                current_node, neighbor, congestion_tier
            ]

            heuristic_bonus = 0.0

            if (
                edge_heuristics is not None
                and neighbor in edge_heuristics
            ):
                heuristic_bonus = float(edge_heuristics[neighbor])

                if not np.isfinite(heuristic_bonus):
                    raise ValueError(
                        "Edge heuristic values must be finite."
                    )

            total_value = q_value + heuristic_bonus

            if total_value > best_value:
                best_value = total_value
                best_node = neighbor

        return int(best_node)

    def store_transition(
        self,
        state_tuple,
        action,
        reward,
        next_state_tuple,
        done,
    ):
        """Store one transition in the bounded replay buffer."""
        if len(state_tuple) != 2 or len(next_state_tuple) != 2:
            raise ValueError(
                "State tuples must be (node, congestion_tier)."
            )

        current_node = self._validate_node(state_tuple[0])
        current_tier = self._validate_tier(state_tuple[1])
        action = self._validate_node(action)

        next_node = self._validate_node(next_state_tuple[0])
        next_tier = self._validate_tier(next_state_tuple[1])

        reward = float(reward)

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be a boolean.")

        self.replay_buffer.append(
            (
                (current_node, current_tier),
                action,
                reward,
                (next_node, next_tier),
                bool(done),
            )
        )

    def update_policy(self, batch_size=32):
        """
        Update Q-values using sampled one-step TD targets.

        Returns:
            Mean absolute TD error, or 0.0 if no transitions exist.
        """
        if not isinstance(batch_size, (int, np.integer)):
            raise TypeError("batch_size must be an integer.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        buffer_size = len(self.replay_buffer)

        if buffer_size == 0:
            return 0.0

        sample_size = min(buffer_size, int(batch_size))

        indices = self.rng.choice(
            buffer_size,
            size=sample_size,
            replace=False,
        )

        td_losses = []

        for idx in indices:
            (
                (curr_node, curr_tier),
                action,
                reward,
                (next_node, next_tier),
                done,
            ) = self.replay_buffer[int(idx)]

            current_q = self.q_table[
                curr_node, action, curr_tier
            ]

            if done:
                target_q = reward
            else:
                target_q = (
                    reward
                    + self.gamma
                    * np.max(
                        self.q_table[next_node, :, next_tier]
                    )
                )

            td_error = target_q - current_q

            self.q_table[
                curr_node, action, curr_tier
            ] += self.lr * td_error

            td_losses.append(abs(td_error))

        return float(np.mean(td_losses))

    def decay_exploration(self):
        """Reduce epsilon after each training episode."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )
