import os
import csv
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from envs.env_1uav_5user_power import UAV6GMultiUserEnv


# ============================================================
# Configuration
# ============================================================

TRAINING_SEEDS = [0, 42, 123]
EVALUATION_SEED = 42
SCENARIOS = ["evacuation", "straggler"]
EVALUATION_RHO = 0.1
MAX_STEPS = 200

MODEL_NAME_TEMPLATE = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_smooth_1m_seed{seed}"
)

OUTPUT_DIR = os.path.join(
    "./evaluations",
    "training_seed_robustness",
    f"eval_seed{EVALUATION_SEED}",
)


# ============================================================
# Single-model lightweight evaluation
# ============================================================

def evaluate_single_model(
    model_path: str,
    scenario: str,
    evaluation_seed: int = 42,
    rho: float = 0.1,
    max_steps: int = 200,
) -> Dict[str, float]:
    """
    Evaluate one trained Group-A model under a fixed evaluation
    environment.

    Only the training seed differs across models. The evaluation
    seed, scenario settings, interference coefficient, and episode
    length are kept fixed.
    """

    if (
        not os.path.exists(model_path)
        and not os.path.exists(model_path + ".zip")
    ):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip"
        )

    env = UAV6GMultiUserEnv(
        num_users=5,
        mode="general",
        scenario=scenario,
        use_interference=True,
        interference_coupling=rho,
        power_allocation_mode="learned",
    )

    model = PPO.load(
        model_path,
        device="cpu",
    )

    obs, _ = env.reset(seed=evaluation_seed)

    ee_history: List[float] = []
    min_rate_history: List[float] = []
    avg_rate_history: List[float] = []
    jain_history: List[float] = []
    qos_history: List[float] = []

    for _ in range(max_steps):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, _, terminated, truncated, info = env.step(action)

        ee_history.append(
            float(info["energy_efficiency"]) / 1e6
        )
        min_rate_history.append(
            float(info["min_rate_mbps"])
        )
        avg_rate_history.append(
            float(info["avg_rate_mbps"])
        )
        jain_history.append(
            float(info["jain_index"])
        )
        qos_history.append(
            float(info["qos_violation_ratio"])
        )

        if terminated or truncated:
            break

    env.close()

    if not ee_history:
        raise RuntimeError(
            f"No evaluation steps produced for {model_path}, "
            f"scenario={scenario}"
        )

    return {
        "mean_ee_mbit_j": float(np.mean(ee_history)),
        "mean_min_rate_mbps": float(np.mean(min_rate_history)),
        "mean_avg_rate_mbps": float(np.mean(avg_rate_history)),
        "mean_jain": float(np.mean(jain_history)),
        "mean_qos_violation": float(np.mean(qos_history)),
        "episode_steps": int(len(ee_history)),
    }


# ============================================================
# Multi-training-seed evaluation
# ============================================================

def run_training_seed_robustness():
    """
    Evaluate Group-A models trained with seeds 0, 42, and 123.

    All models are evaluated using:
        evaluation seed = 42
        rho = 0.1
        deterministic policy
        200-step maximum episode
        evacuation and straggler scenarios

    Outputs:
        1. training_seed_metrics.csv
        2. training_seed_summary.csv
        3. training_seed_robustness.npy
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    print("\n" + "=" * 76)
    print("Training-seed robustness evaluation")
    print("=" * 76)
    print(f"Training seeds: {TRAINING_SEEDS}")
    print(f"Fixed evaluation seed: {EVALUATION_SEED}")
    print(f"Fixed evaluation rho: {EVALUATION_RHO}")
    print(f"Maximum evaluation steps: {MAX_STEPS}")
    print("=" * 76)

    for scenario in SCENARIOS:
        print("\n" + "-" * 76)
        print(f"Scenario: {scenario}")
        print("-" * 76)

        for training_seed in TRAINING_SEEDS:
            experiment_name = MODEL_NAME_TEMPLATE.format(
                seed=training_seed
            )

            model_path = os.path.join(
                "./models",
                experiment_name,
                "model",
            )

            print(f"\nTraining seed = {training_seed}")
            print(f"Model: {model_path}.zip")

            result = evaluate_single_model(
                model_path=model_path,
                scenario=scenario,
                evaluation_seed=EVALUATION_SEED,
                rho=EVALUATION_RHO,
                max_steps=MAX_STEPS,
            )

            row = {
                "scenario": scenario,
                "training_seed": int(training_seed),
                "evaluation_seed": int(EVALUATION_SEED),
                "rho": float(EVALUATION_RHO),
                "mean_ee_mbit_j": result["mean_ee_mbit_j"],
                "mean_min_rate_mbps": result["mean_min_rate_mbps"],
                "mean_avg_rate_mbps": result["mean_avg_rate_mbps"],
                "mean_jain": result["mean_jain"],
                "mean_qos_violation": result["mean_qos_violation"],
                "episode_steps": result["episode_steps"],
            }

            all_results.append(row)

            print(
                f"EE={row['mean_ee_mbit_j']:.2f} Mbit/J | "
                f"MinRate={row['mean_min_rate_mbps']:.2f} Mbps | "
                f"AvgRate={row['mean_avg_rate_mbps']:.2f} Mbps | "
                f"Jain={row['mean_jain']:.4f} | "
                f"QoS={row['mean_qos_violation']:.4f}"
            )

    metric_keys = {
        "EE (Mbit/J)": "mean_ee_mbit_j",
        "Minimum Rate (Mbps)": "mean_min_rate_mbps",
        "Average Rate (Mbps)": "mean_avg_rate_mbps",
        "Jain Index": "mean_jain",
        "QoS Violation Ratio": "mean_qos_violation",
    }

    summary_results = {}

    print("\n" + "=" * 76)
    print("Random-seed robustness summary")
    print("Values are mean ± sample standard deviation")
    print("=" * 76)

    for scenario in SCENARIOS:
        scenario_rows = [
            row for row in all_results
            if row["scenario"] == scenario
        ]

        summary_results[scenario] = {}
        print(f"\n{scenario.upper()}")

        for display_name, metric_key in metric_keys.items():
            values = np.asarray(
                [row[metric_key] for row in scenario_rows],
                dtype=np.float64,
            )

            metric_mean = float(np.mean(values))
            metric_std = float(np.std(values, ddof=1))

            summary_results[scenario][metric_key] = {
                "mean": metric_mean,
                "std": metric_std,
                "values": values,
            }

            print(
                f"{display_name}: "
                f"{metric_mean:.4f} ± {metric_std:.4f}"
            )

    raw_csv_path = os.path.join(
        OUTPUT_DIR,
        "training_seed_metrics.csv",
    )

    raw_fieldnames = [
        "scenario",
        "training_seed",
        "evaluation_seed",
        "rho",
        "mean_ee_mbit_j",
        "mean_min_rate_mbps",
        "mean_avg_rate_mbps",
        "mean_jain",
        "mean_qos_violation",
        "episode_steps",
    ]

    with open(
        raw_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=raw_fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_results)

    summary_csv_path = os.path.join(
        OUTPUT_DIR,
        "training_seed_summary.csv",
    )

    with open(
        summary_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "scenario",
                "metric",
                "mean",
                "std",
                "mean_plus_minus_std",
                "n_training_seeds",
            ]
        )

        for scenario in SCENARIOS:
            for display_name, metric_key in metric_keys.items():
                stats = summary_results[scenario][metric_key]
                writer.writerow(
                    [
                        scenario,
                        display_name,
                        stats["mean"],
                        stats["std"],
                        f"{stats['mean']:.4f} ± {stats['std']:.4f}",
                        len(TRAINING_SEEDS),
                    ]
                )

    npy_path = os.path.join(
        OUTPUT_DIR,
        "training_seed_robustness.npy",
    )

    np.save(
        npy_path,
        {
            "training_seeds": np.asarray(TRAINING_SEEDS),
            "evaluation_seed": EVALUATION_SEED,
            "rho": EVALUATION_RHO,
            "max_steps": MAX_STEPS,
            "scenarios": np.asarray(SCENARIOS),
            "all_results": all_results,
            "summary_results": summary_results,
        },
        allow_pickle=True,
    )

    print("\n" + "=" * 76)
    print("Generated files")
    print("=" * 76)
    print(f"- {raw_csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {npy_path}")

    return {
        "all_results": all_results,
        "summary_results": summary_results,
    }


if __name__ == "__main__":
    run_training_seed_robustness()