import numpy as np


class IoTNetworkSimulator:
    def __init__(self, max_range_m=500.0, seed=42):
        if not np.isfinite(max_range_m) or max_range_m <= 0:
            raise ValueError("max_range_m must be a finite positive number")
        self.max_range_m = float(max_range_m)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _finite(name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def compute_blockchain_auth_delay(self, num_drones=20, t_hash_ms=12.0, t_verify_ms=28.0):
        """Compute modeled blockchain authentication delay in milliseconds."""
        if num_drones < 1:
            raise ValueError("num_drones must be at least 1")
        t_hash_ms = self._finite("t_hash_ms", t_hash_ms)
        t_verify_ms = self._finite("t_verify_ms", t_verify_ms)
        if t_hash_ms < 0 or t_verify_ms < 0:
            raise ValueError("hash and verification times must be nonnegative")
        delay = (t_hash_ms + t_verify_ms) / num_drones + 0.15 * t_verify_ms
        return float(max(2.0, delay))

    def compute_sensor_fusion(self, primary_score, failure_prob=0.06, weight=0.82):
        """Fuse a primary score with its failure-adjusted estimate."""
        primary_score = self._finite("primary_score", primary_score)
        failure_prob = self._finite("failure_prob", failure_prob)
        weight = self._finite("weight", weight)
        if not 0.0 <= failure_prob <= 1.0 or not 0.0 <= weight <= 1.0:
            raise ValueError("failure_prob and weight must be in [0, 1]")
        score = np.clip(primary_score, 0.0, 1.0)
        aggregate = weight * score + (1.0 - weight) * (1.0 - failure_prob) * score
        return float(np.clip(aggregate, 0.0, 1.0))

    def filter_sensor_outliers(self, raw_readings, k_threshold=1.6):
        """Replace readings outside k standard deviations with the mean."""
        values = np.asarray(raw_readings, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("raw_readings must be one-dimensional")
        if len(values) == 0:
            return 0.5
        if not np.all(np.isfinite(values)):
            raise ValueError("raw_readings must contain only finite values")
        k_threshold = self._finite("k_threshold", k_threshold)
        if k_threshold <= 0:
            raise ValueError("k_threshold must be positive")
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return float(mean)
        filtered = np.where(np.abs(values - mean) <= k_threshold * std, values, mean)
        return float(np.mean(filtered))

    def compute_edge_cloud_latency(
        self, comp_load=0.40, f_edge_ghz=2.5, tau_cache_ms=5.0,
        t_up_ms=9.0, t_pro_ms=14.0, t_down_ms=7.0
    ):
        """Return modeled edge-processing and cloud-sync latency in ms."""
        values = {
            "comp_load": comp_load, "f_edge_ghz": f_edge_ghz,
            "tau_cache_ms": tau_cache_ms, "t_up_ms": t_up_ms,
            "t_pro_ms": t_pro_ms, "t_down_ms": t_down_ms,
        }
        values = {k: self._finite(k, v) for k, v in values.items()}
        if values["comp_load"] < 0 or values["f_edge_ghz"] <= 0:
            raise ValueError("comp_load must be nonnegative and f_edge_ghz positive")
        if any(values[k] < 0 for k in ("tau_cache_ms", "t_up_ms", "t_pro_ms", "t_down_ms")):
            raise ValueError("latency components must be nonnegative")
        f_norm = np.clip(values["f_edge_ghz"] / 5.0, 0.15, 0.95)
        t_edge = (1.0 - f_norm * 0.4) * (values["comp_load"] / f_norm) * 18.0 + values["tau_cache_ms"]
        t_up_norm = np.clip(values["t_up_ms"] / 80.0, 0.05, 0.6)
        t_sync = values["t_up_ms"] + values["t_pro_ms"] + values["t_down_ms"] * (1.0 - 0.5 * t_up_norm)
        return float(t_edge), float(t_sync)

    def compute_channel_strength(self, latency_norm, noise_var=0.07, e_enc=0.97, p_auth=0.98):
        """Compute bounded secure-channel strength from normalized inputs."""
        latency_norm = self._finite("latency_norm", latency_norm)
        noise_var = self._finite("noise_var", noise_var)
        e_enc = self._finite("e_enc", e_enc)
        p_auth = self._finite("p_auth", p_auth)
        if latency_norm < 0 or noise_var < 0 or not 0 <= e_enc <= 1 or not 0 <= p_auth <= 1:
            raise ValueError("latency/noise must be nonnegative and encryption/authentication in [0,1]")
        denom = latency_norm + noise_var + 0.12
        strength = ((e_enc * p_auth) / denom) * (1.0 - noise_var * latency_norm)
        return float(np.clip(strength, 0.0, 1.0))

    def compute_qos_index(self, r_link, eta_lat, a_blk=0.97, w1=0.40, w2=0.35, w3=0.25):
        """Compute a weighted, bounded QoS score."""
        scores = [self._finite(n, v) for n, v in (("r_link", r_link), ("eta_lat", eta_lat), ("a_blk", a_blk))]
        weights = [self._finite(n, v) for n, v in (("w1", w1), ("w2", w2), ("w3", w3))]
        if any(not 0 <= v <= 1 for v in scores + weights):
            raise ValueError("QoS scores and weights must be in [0, 1]")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("QoS weights must sum to 1")
        return float(np.clip(sum(s * w for s, w in zip(scores, weights)), 0.0, 1.0))

    def compute_secure_throughput(self, data_mb, packet_loss, comm_time_s):
        """Return secure throughput in Mbps (decimal megabits per second)."""
        data_mb = self._finite("data_mb", data_mb)
        packet_loss = self._finite("packet_loss", packet_loss)
        comm_time_s = self._finite("comm_time_s", comm_time_s)
        if data_mb < 0 or not 0 <= packet_loss <= 1 or comm_time_s < 0:
            raise ValueError("data_mb and comm_time_s must be nonnegative; packet_loss must be in [0,1]")
        if comm_time_s == 0:
            return 0.0
        # 1 MB = 8 megabits; apply packet-loss and secure-channel overhead.
        throughput = (data_mb * 8.0 * (1.0 - packet_loss) / comm_time_s) * (1.0 - 0.12 * packet_loss)
        return float(max(0.0, throughput))

    def compute_operational_security(self, q_svc, r_link, t_sec_norm, comm_time_norm=0.22):
        """Compute the bounded operational security score."""
        q_svc, r_link, t_sec_norm, comm_time_norm = [
            self._finite(n, v) for n, v in (
                ("q_svc", q_svc), ("r_link", r_link),
                ("t_sec_norm", t_sec_norm), ("comm_time_norm", comm_time_norm)
            )
        ]
        if any(not 0 <= v <= 1 for v in (q_svc, r_link, t_sec_norm, comm_time_norm)):
            raise ValueError("security score inputs must be in [0, 1]")
        raw = ((q_svc + r_link + t_sec_norm) / 3.0) * (1.0 - 0.08 * comm_time_norm * q_svc)
        return float(np.clip(raw, 0.0, 1.0))

    def simulate_wireless_link(
        self, distance_m, congestion_level, num_drones=20,
        is_rl_aro=None, payload_mb=60.0
    ):
        """Simulate one link and return metrics compatible with the evaluator.

        `is_rl_aro` is retained for source compatibility with the earlier
        draft, but intentionally has no effect: policy identity must not
        artificially improve outcomes. Supply actual route distance and
        congestion to compare methods fairly.
        """
        distance_m = self._finite("distance_m", distance_m)
        payload_mb = self._finite("payload_mb", payload_mb)
        if distance_m < 0 or payload_mb < 0:
            raise ValueError("distance_m and payload_mb must be nonnegative")
        if num_drones < 1:
            raise ValueError("num_drones must be at least 1")

        dist_ratio = np.clip(distance_m / self.max_range_m, 0.05, 1.3)
        congestion = float(np.clip(self._finite("congestion_level", congestion_level), 0.0, 1.0))

        raw_samples = [congestion + self.rng.normal(0, 0.035) for _ in range(6)]
        clean_congestion = float(np.clip(self.filter_sensor_outliers(raw_samples), 0.0, 1.0))

        packet_loss = float(np.clip(
            0.018 + 0.055 * dist_ratio + 0.070 * clean_congestion + self.rng.uniform(0.0, 0.025),
            0.008, 0.22
        ))
        pdr_percent = (1.0 - packet_loss) * 100.0
        r_link = 1.0 - packet_loss

        t_edge, t_sync = self.compute_edge_cloud_latency(
            comp_load=0.32 + 0.35 * clean_congestion
        )
        auth_delay_ms = self.compute_blockchain_auth_delay(num_drones=num_drones)
        propagation_ms = 12.0 * dist_ratio + 22.0 * clean_congestion
        total_latency_ms = float(np.clip(
            t_edge + t_sync + propagation_ms + auth_delay_ms, 8.0, 180.0
        ))
        latency_norm = float(np.clip(total_latency_ms / 140.0, 0.05, 0.95))
        eta_lat = 1.0 - latency_norm

        channel_strength = self.compute_channel_strength(latency_norm)
        qos = self.compute_qos_index(r_link, eta_lat)
        throughput = self.compute_secure_throughput(payload_mb, packet_loss, comm_time_s=0.90)
        throughput_norm = float(np.clip(throughput / 70.0, 0.0, 1.0))
        operational_security = self.compute_operational_security(qos, r_link, throughput_norm)
        aggregate_score = self.compute_sensor_fusion(qos)

        return {
            "distance_m": float(distance_m),
            "packet_loss": packet_loss,
            "pdr_percent": float(pdr_percent),
            "latency_ms": total_latency_ms,
            "throughput_mbps": float(throughput),
            "b_auth_ms": float(auth_delay_ms),
            "s_ch": float(channel_strength),
            "q_svc": float(qos),
            "s_ops": float(operational_security),
            "d_agg": float(aggregate_score),
            "r_link": float(r_link),
            "congestion_clean": clean_congestion,
        }
