import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class RLAROEvaluator:
    def __init__(self, seeds=(42, 123, 256, 512, 1024)):
        self.seeds = list(seeds)
        if len(self.seeds) < 2:
            raise ValueError("At least two seeds are required for summary statistics")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")

    # ------------------------------------------------------------------
    # Metric calculations (aligned with paper equations)
    # ------------------------------------------------------------------

    def time_optimization_gain(self, adt_base, adt_rl):
        """Delivery time reduction percentage (inspired by Eq. 13)."""
        if adt_base <= 0:
            return 0.0
        return float(((adt_base - adt_rl) / adt_base) * 100.0)

    def energy_saving_potential(self, tec_base, tec_rl):
        """Energy saving percentage (inspired by Eq. 14)."""
        if tec_base <= 0:
            return 0.0
        return float(((tec_base - tec_rl) / tec_base) * 100.0)

    def paired_ttest(self, baseline_samples, rl_samples):
        """Paired two-tailed Student t-test (paper Section 5.1)."""
        baseline_samples = np.asarray(baseline_samples, dtype=np.float64)
        rl_samples = np.asarray(rl_samples, dtype=np.float64)

        if baseline_samples.ndim != 1 or rl_samples.ndim != 1:
            raise ValueError("Paired samples must be one-dimensional")
        if len(baseline_samples) != len(rl_samples):
            raise ValueError("Sample lists must have the same length")
        if len(baseline_samples) < 2:
            raise ValueError("At least two paired observations are required")
        if not np.all(np.isfinite(baseline_samples)) or not np.all(np.isfinite(rl_samples)):
            raise ValueError("Paired samples must contain only finite values")
        if np.array_equal(baseline_samples, rl_samples):
            return 0.0, 1.0

        result = stats.ttest_rel(baseline_samples, rl_samples)
        return float(result.statistic), float(result.pvalue)

    def summarize(self, values):
        """Return mean and standard deviation."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("values must be a one-dimensional array with at least two samples")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must contain only finite numbers")
        return float(np.mean(values)), float(np.std(values, ddof=1))

    # ------------------------------------------------------------------
    # Main evaluation from raw seed results
    # ------------------------------------------------------------------

    def evaluate(self, seed_results):
        """
        seed_results: list of per-seed aggregate dictionaries, one per seed,
        in the same seed order for all methods (required for paired tests).
        Aggregate route/simulation outputs from the environment and agent
        before calling this method; the environment does not produce PDR or
        latency values by itself. Each dict must contain:
            {
                "ACO": {"delivery_time_min": ..., "fleet_energy_wh": ..., "pdr_percent": ..., "latency_ms": ...},
                "TSP-D": {...},
                "DBSCAN": {...},
                "RL-ARO": {...},
                "rl_path": [...]
            }
        """
        methods = ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]
        metrics = ["delivery_time_min", "fleet_energy_wh", "pdr_percent", "latency_ms"]
        if not isinstance(seed_results, (list, tuple)) or len(seed_results) != len(self.seeds):
            raise ValueError(f"seed_results must contain exactly {len(self.seeds)} per-seed records")

        # Collect samples
        data = {m: {met: [] for met in metrics} for m in methods}

        for seed_index, res in enumerate(seed_results):
            if not isinstance(res, dict):
                raise ValueError(f"result for seed index {seed_index} must be a dictionary")
            for method in methods:
                if method not in res or not isinstance(res[method], dict):
                    raise ValueError(f"seed result {seed_index} is missing method {method!r}")
                for metric in metrics:
                    if metric not in res[method]:
                        raise ValueError(f"seed result {seed_index}, method {method!r} is missing {metric!r}")
                    value = float(res[method][metric])
                    if not np.isfinite(value):
                        raise ValueError(f"{method}.{metric} must be finite")
                    if metric == "pdr_percent" and not 0.0 <= value <= 100.0:
                        raise ValueError(f"{method}.pdr_percent must be between 0 and 100")
                    if metric != "pdr_percent" and value < 0.0:
                        raise ValueError(f"{method}.{metric} must be nonnegative")
                    data[method][metric].append(value)

        # Compute summary statistics
        summary = {}
        for m in methods:
            summary[m] = {}
            for met in metrics:
                mean, std = self.summarize(data[m][met])
                summary[m][met] = {"mean": mean, "std": std, "samples": data[m][met]}

        # Statistical tests (RL-ARO vs each baseline on delivery time)
        ttests = {}
        for baseline in ["ACO", "TSP-D", "DBSCAN"]:
            t_stat, p_val = self.paired_ttest(
                data[baseline]["delivery_time_min"],
                data["RL-ARO"]["delivery_time_min"]
            )
            ttests[baseline] = {"t_statistic": t_stat, "p_value": p_val}

        # Key derived metrics
        tog_vs_aco = self.time_optimization_gain(
            summary["ACO"]["delivery_time_min"]["mean"],
            summary["RL-ARO"]["delivery_time_min"]["mean"]
        )
        esp_vs_aco = self.energy_saving_potential(
            summary["ACO"]["fleet_energy_wh"]["mean"],
            summary["RL-ARO"]["fleet_energy_wh"]["mean"]
        )

        return {
            "summary": summary,
            "ttests": ttests,
            "time_reduction_vs_ACO_percent": tog_vs_aco,
            "energy_saving_vs_ACO_percent": esp_vs_aco,
            "raw_data": data
        }

    # ------------------------------------------------------------------
    # Plot generation from real data
    # ------------------------------------------------------------------

    def generate_dashboard(self, eval_result, seed_results, output_path="RL_ARO_Simulation_Dashboard.png"):
        """
        Creates a clean dashboard using only real simulation results.
        """
        summary = eval_result["summary"]
        methods = ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]
        colors = {
            "ACO": "#d97706",
            "TSP-D": "#15803d",
            "DBSCAN": "#2563eb",
            "RL-ARO": "#7c3aed"
        }

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("RL-ARO Evaluation Dashboard (Real Simulation Results)", fontsize=15, fontweight="bold")

        # 1. Delivery Time comparison (bar)
        ax = axes[0, 0]
        means = [summary[m]["delivery_time_min"]["mean"] for m in methods]
        stds = [summary[m]["delivery_time_min"]["std"] for m in methods]
        bars = ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], alpha=0.85, capsize=5)
        ax.set_ylabel("Delivery Time (min)")
        ax.set_title("Average Delivery Time")
        ax.grid(True, axis="y", alpha=0.3)

        # 2. Energy comparison
        ax = axes[0, 1]
        means = [summary[m]["fleet_energy_wh"]["mean"] for m in methods]
        stds = [summary[m]["fleet_energy_wh"]["std"] for m in methods]
        ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], alpha=0.85, capsize=5)
        ax.set_ylabel("Fleet Energy (Wh)")
        ax.set_title("Average Fleet Energy Consumption")
        ax.grid(True, axis="y", alpha=0.3)

        # 3. Packet Delivery Ratio
        ax = axes[0, 2]
        means = [summary[m]["pdr_percent"]["mean"] for m in methods]
        stds = [summary[m]["pdr_percent"]["std"] for m in methods]
        ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], alpha=0.85, capsize=5)
        ax.set_ylabel("PDR (%)")
        ax.set_title("Packet Delivery Ratio")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)

        # 4. Latency
        ax = axes[1, 0]
        means = [summary[m]["latency_ms"]["mean"] for m in methods]
        stds = [summary[m]["latency_ms"]["std"] for m in methods]
        ax.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], alpha=0.85, capsize=5)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Average Communication Latency")
        ax.grid(True, axis="y", alpha=0.3)

        # 5. Delivery time reduction vs baselines
        ax = axes[1, 1]
        reductions = []
        labels = []
        for b in ["ACO", "TSP-D", "DBSCAN"]:
            red = self.time_optimization_gain(
                summary[b]["delivery_time_min"]["mean"],
                summary["RL-ARO"]["delivery_time_min"]["mean"]
            )
            reductions.append(red)
            labels.append(f"vs {b}")
        ax.bar(labels, reductions, color="#7c3aed", alpha=0.85)
        ax.set_ylabel("Reduction (%)")
        ax.set_title("RL-ARO Delivery Time Reduction")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        # 6. Statistical significance (p-values)
        ax = axes[1, 2]
        pvals = [eval_result["ttests"][b]["p_value"] for b in ["ACO", "TSP-D", "DBSCAN"]]
        bars = ax.bar(["vs ACO", "vs TSP-D", "vs DBSCAN"], pvals, color="#dc2626", alpha=0.8)
        ax.axhline(0.05, color="green", linestyle="--", label="α = 0.05")
        ax.set_ylabel("p-value")
        ax.set_title("Paired t-test p-values (Delivery Time)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Pretty printing of results
    # ------------------------------------------------------------------

    def print_report(self, eval_result, rl_path=None):
        summary = eval_result["summary"]
        print("=" * 70)
        print(" RL-ARO EVALUATION REPORT (Real Simulation Results)")
        print("=" * 70)

        if rl_path is not None:
            print(f"Learned RL-ARO Route: {rl_path}")

        print("\n--- Average Delivery Time (minutes) ---")
        for m in ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]:
            mean = summary[m]["delivery_time_min"]["mean"]
            std = summary[m]["delivery_time_min"]["std"]
            print(f"{m:8s}: {mean:7.2f} ± {std:.2f}")

        print(f"\nDelivery Time Reduction vs ACO: {eval_result['time_reduction_vs_ACO_percent']:.2f}%")

        print("\n--- Fleet Energy (Wh) ---")
        for m in ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]:
            mean = summary[m]["fleet_energy_wh"]["mean"]
            std = summary[m]["fleet_energy_wh"]["std"]
            print(f"{m:8s}: {mean:8.2f} ± {std:.2f}")

        print(f"\nEnergy Saving vs ACO: {eval_result['energy_saving_vs_ACO_percent']:.2f}%")

        print("\n--- Packet Delivery Ratio (%) ---")
        for m in ["ACO", "TSP-D", "DBSCAN", "RL-ARO"]:
            mean = summary[m]["pdr_percent"]["mean"]
            std = summary[m]["pdr_percent"]["std"]
            print(f"{m:8s}: {mean:6.2f} ± {std:.2f}")

        print("\n--- Statistical Significance (Paired t-test on Delivery Time) ---")
        for b, res in eval_result["ttests"].items():
            sig = "YES" if res["p_value"] < 0.05 else "NO"
            print(f"RL-ARO vs {b:7s}: t = {res['t_statistic']:7.4f}, p = {res['p_value']:.5f}  → Significant: {sig}")

        print("=" * 70)