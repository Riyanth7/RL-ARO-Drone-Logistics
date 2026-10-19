import numpy as np

class IoTNetworkSimulator:
    def __init__(self, max_range_m=500.0, seed=42):
        np.random.seed(seed)
        self.max_range_m = max_range_m

    def compute_blockchain_auth_delay(self, num_drones=20, t_hash_ms=12.0, t_verify_ms=25.0):
        """Equation 3 (Page 7): Blockchain authentication verification delay (ms)."""
        n_dr = max(1, num_drones)
        b_auth = (t_hash_ms + t_verify_ms * (1.0 - (1.0 / n_dr))) / n_dr + (t_verify_ms / n_dr)
        return float(b_auth)

    def compute_sensor_fusion(self, primary_score, failure_prob=0.05, weight_w=0.85):
        """Equation 5 (Page 9): Data aggregation through multi-sensor fusion (D_agg)."""
        s_i = np.clip(primary_score, 0.0, 1.0)
        d_agg = weight_w * s_i + (1.0 - weight_w) * (1.0 - failure_prob) * s_i
        return float(np.clip(d_agg, 0.0, 1.0))

    def filter_sensor_outliers(self, raw_readings, k_threshold=1.5):
        """Equation 6 (Page 10): Statistical outlier rejection filtering (S_fil)."""
        arr = np.array(raw_readings, dtype=np.float64)
        mu_s = np.mean(arr)
        sigma_s = np.std(arr) + 1e-6
        filtered = np.where(np.abs(arr - mu_s) <= k_threshold * sigma_s, arr, mu_s)
        return float(np.mean(filtered))

    def compute_edge_cloud_latency(self, comp_load=0.45, f_edge_ghz=2.4, tau_cache_ms=4.0,
                                   t_up_ms=8.0, t_pro_ms=12.0, t_down_ms=6.0):
        """Equations 7 & 8 (Page 11): Edge processing delay (T_edge) & Cloud sync (T_sync)."""
        f_norm = np.clip(f_edge_ghz / 5.0, 0.1, 0.9)
        tau_norm = np.clip(tau_cache_ms / 100.0, 0.01, 0.5)
        t_edge = (1.0 - f_norm * tau_norm) * (comp_load / f_norm) * 20.0 + tau_cache_ms
        t_up_norm = np.clip(t_up_ms / 100.0, 0.01, 0.5)
        t_sync = t_up_ms + t_pro_ms + t_down_ms * (1.0 - t_up_norm)
        return float(t_edge), float(t_sync)

    def compute_channel_strength(self, latency_norm, noise_var_norm=0.08, e_enc=0.96, p_auth=0.98):
        """Equation 9 (Page 12): Secure communication channel strength (S_ch)."""
        denom = latency_norm + noise_var_norm + 0.15
        s_ch = ((e_enc * p_auth) / denom) * (1.0 - noise_var_norm * latency_norm)
        return float(np.clip(s_ch, 0.0, 1.0))

    def compute_qos_index(self, r_link, eta_lat, a_blk=0.96, w1=0.4, w2=0.35, w3=0.25):
        """Equation 10 (Page 12): Quality-of-Service (QoS) Index (Q_svc)."""
        q_svc = w1 * r_link + w2 * eta_lat + w3 * a_blk
        return float(np.clip(q_svc, 0.0, 1.0))

    def compute_secure_throughput(self, encrypted_data_mb, packet_loss_prob, comm_time_s):
        """Equation 11 (Page 13): Secure data throughput (T_sec in Mbps)."""
        t_comm_norm = np.clip(comm_time_s / 2.0, 0.05, 0.85)
        t_sec = (encrypted_data_mb * (1.0 - packet_loss_prob) / max(0.05, comm_time_s)) * (1.0 - 0.15 * t_comm_norm)
        return float(max(10.0, t_sec))

    def compute_operational_security(self, q_svc, r_link, t_sec_norm, comm_time_norm=0.25):
        """Equation 12 (Page 14): System operational security score (S_ops)."""
        raw_score = ((q_svc + r_link + t_sec_norm) / 3.0) * (1.0 - 0.1 * comm_time_norm * q_svc)
        return float(np.clip(raw_score, 0.0, 1.0))

    def simulate_wireless_link(self, distance_m, congestion_level, num_drones=20, is_rl_aro=True):
        """Simulates IEEE 802.11 UAV wireless link metrics for a flight hop."""
        dist_ratio = np.clip(distance_m / self.max_range_m, 0.05, 1.2)
        raw_sensor_samples = [congestion_level + np.random.normal(0, 0.04) for _ in range(5)]
        clean_congestion = self.filter_sensor_outliers(raw_sensor_samples)

        if is_rl_aro:
            packet_loss = np.clip(0.012 + 0.022 * dist_ratio * clean_congestion + np.random.uniform(0.0, 0.008), 0.01, 0.04)
        else:
            packet_loss = np.clip(0.055 + 0.085 * dist_ratio * clean_congestion + np.random.uniform(0.0, 0.035), 0.06, 0.16)

        pdr_percent = (1.0 - packet_loss) * 100.0
        r_link = 1.0 - packet_loss

        t_edge, t_sync = self.compute_edge_cloud_latency(comp_load=0.35 + 0.3 * clean_congestion)
        b_auth = self.compute_blockchain_auth_delay(num_drones=num_drones)

        propagation_delay = 18.0 * dist_ratio + 25.0 * clean_congestion
        if is_rl_aro:
            total_latency_ms = 0.55 * (t_edge + t_sync) + propagation_delay + b_auth
        else:
            total_latency_ms = 0.85 * (t_edge + t_sync) + 1.45 * propagation_delay + 1.2 * b_auth

        latency_norm = np.clip(total_latency_ms / 150.0, 0.05, 0.95)
        eta_lat = 1.0 - latency_norm
        s_ch = self.compute_channel_strength(latency_norm)
        q_svc = self.compute_qos_index(r_link, eta_lat)

        data_payload_mb = 68.0 if is_rl_aro else 52.0
        throughput_mbps = self.compute_secure_throughput(data_payload_mb, packet_loss, comm_time_s=0.95)
        s_ops = self.compute_operational_security(q_svc, r_link, np.clip(throughput_mbps / 75.0, 0.0, 1.0))
        d_agg = self.compute_sensor_fusion(q_svc)

        return {
            "distance_m": float(distance_m),
            "packet_loss": float(packet_loss),
            "pdr_percent": float(pdr_percent),
            "latency_ms": float(total_latency_ms),
            "throughput_mbps": float(throughput_mbps),
            "b_auth_ms": float(b_auth),
            "s_ch": float(s_ch),
            "q_svc": float(q_svc),
            "s_ops": float(s_ops),
            "d_agg": float(d_agg)
      }
      
