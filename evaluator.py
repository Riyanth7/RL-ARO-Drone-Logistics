import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class RLAROEvaluator:
    def __init__(self, seeds=(42, 123, 256, 512, 1024)):
        self.seeds = seeds

    def compute_time_optimization_gain(self, adt_base, adt_rl):
        """Equation 13 (Page 18): Percentage gain in delivery time reduction (TOG %)."""
        return float(((adt_base - adt_rl) / max(1e-6, adt_base)) * 100.0)

    def compute_energy_saving_potential(self, tec_base, tec_rl):
        """Equation 14 (Page 19): Energy Saving Potential percentage (ESP %)."""
        return float(((tec_base - tec_rl) / max(1e-6, tec_base)) * 100.0)

    def compute_reward_gain_throughput(self, base_reward, path_deviation_error, alpha=0.25):
        """Equation 15 (Page 21): System throughput reward gain penalty (RRG_t)."""
        return float(base_reward - alpha * path_deviation_error)

    def compute_route_optimality_index(self, gamma_s=0.92, tau_e=0.95, delta_n=0.07):
        """Equation 16 (Page 21): Adaptive delivery strength & Route Optimality Index (%)."""
        psi_2 = ((gamma_s * tau_e) / (gamma_s * tau_e + delta_n)) * (1.0 - 0.02 * delta_n)
        return float(np.clip(psi_2 * 100.0, 70.0, 96.0))

    def compute_adaptation_response_time(self, t_det_s=1.4, t_pro_s=2.1, t_adj_s=2.2):
        """Equation 17 (Page 22): Adaptation Response Time (ART in seconds)."""
        art = t_det_s + t_pro_s + t_adj_s * (1.0 - 0.1 * t_det_s)
        return float(art)

    def compute_network_resilience_score(self, avail=0.96, rec_time_norm=0.08, red_eff=0.92, cons_stab=0.94):
        """Equation 18 (Page 22): Composite Network Resilience Score (R %)."""
        r_score = ((avail * (1.0 - rec_time_norm) + red_eff + cons_stab) / 3.0) * (1.0 - 0.03 * (1.0 - red_eff))
        return float(np.clip(r_score * 100.0, 50.0, 99.0))

    def compute_convergence_reward(self, delta_acc, delta_loss, alpha=0.2, beta=0.05):
        """Equation 19 (Page 23): Convergence monitoring reward (R_conv)."""
        return float((1.0 - alpha) * (delta_acc - beta) * (1.0 - delta_loss))

    def compute_learning_stability(self, loss_curr, loss_prev):
        """Equation 20 (Page 23): Learning convergence stability metric (S_e)."""
        rel_diff = abs(loss_curr - loss_prev) / max(1e-4, abs(loss_prev))
        s_e = 1.0 - rel_diff * (1.0 / (1.0 + abs(loss_prev)))
        return float(np.clip(s_e, 0.0, 1.0))

    def run_paired_ttest(self, baseline_samples, rl_aro_samples):
        """Section 5.1 (Page 17): Paired two-tailed Student t-test at 95% confidence."""
        t_stat, p_val = stats.ttest_rel(baseline_samples, rl_aro_samples)
        return float(t_stat), float(p_val)

    def generate_all_plots(self, results_dict, output_path="RL_ARO_Simulation_Dashboard.png"):
        """Generates the 5 key comparative charts matching Figures 4, 5, 6, 7, and 8."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
        fig.suptitle("RL-ARO Multi-Agent Drone Logistics Simulation Results (5-Seed Average)",
                     fontsize=15, fontweight="bold")

        styles = {
            "ACO": {"color": "#d97706", "ls": "-", "lw": 1.8},
            "TSP-D": {"color": "#15803d", "ls": "-", "lw": 1.8},
            "DBSCAN": {"color": "#2563eb", "ls": "-", "lw": 1.8},
            "Proposed RL-ARO": {"color": "#581c87", "ls": "--", "lw": 2.5}
        }

        # Plot 1: Fig 4 - Average Delivery Time vs Number of Missions
        ax = axes[0, 0]
        missions = results_dict["missions_x"]
        for name, series in results_dict["fig4_delivery_time"].items():
            ax.plot(missions, series, label=name, **styles[name])
        ax.set_title("Fig 4: Avg Delivery Time vs Missions")
        ax.set_xlabel("Number of Delivery Missions")
        ax.set_ylabel("Average Delivery Time (minutes)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 2: Fig 5 - Energy Consumption vs Fleet Scale
        ax = axes[0, 1]
        drones_x = results_dict["drones_x"]
        for name, series in results_dict["fig5_energy"].items():
            ax.plot(drones_x, series, label=name, **styles[name])
        ax.set_title("Fig 5: Fleet Energy Consumption (Wh)")
        ax.set_xlabel("Active Drone Fleet Scale")
        ax.set_ylabel("Energy Consumption (Wh)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 3: Fig 6 - Route Optimality Index vs Learning Episodes
        ax = axes[0, 2]
        episodes_x = results_dict["episodes_x"]
        for name, series in results_dict["fig6_optimality"].items():
            ax.plot(episodes_x, series, label=name, **styles[name])
        ax.set_title("Fig 6: Route Optimality Index (%)")
        ax.set_xlabel("Learning Episodes")
        ax.set_ylabel("Route Optimality Index (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 4: Fig 7 - Adaptation Response Time vs Environmental Events
        ax = axes[1, 0]
        events_x = results_dict["events_x"]
        for name, series in results_dict["fig7_adaptation"].items():
            ax.plot(events_x, series, label=name, **styles[name])
        ax.set_title("Fig 7: Adaptation Response Time (s)")
        ax.set_xlabel("Environmental Change Events")
        ax.set_ylabel("Response Time (seconds)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 5: Fig 8 - Packet Delivery Ratio vs Comm Attempts
        ax = axes[1, 1]
        comm_x = results_dict["comm_x"]
        for name, series in results_dict["fig8_pdr"].items():
            ax.plot(comm_x, series, label=name, **styles[name])
        ax.set_title("Fig 8: Packet Delivery Ratio (%)")
        ax.set_xlabel("Number of Communication Attempts")
        ax.set_ylabel("Packet Delivery Ratio (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Plot 6: Summary Bar Chart of Table 3 & Table 4 Resilience & Throughput
        ax = axes[1, 2]
        methods = ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]
        throughput_vals = [results_dict["table3"][m]["throughput"] for m in methods]
        resilience_vals = [results_dict["table4"][m]["resilience"] for m in methods]
        x_pos = np.arange(len(methods))
        ax.bar(x_pos - 0.18, throughput_vals, width=0.35, label="Throughput (Mbps)", color="#0284c7")
        ax.bar(x_pos + 0.18, resilience_vals, width=0.35, label="Resilience Score (%)", color="#0d9488")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_title("Tables 3 & 4: Throughput & Resilience")
        ax.set_ylabel("Metric Value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
      
