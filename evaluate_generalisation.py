import os
import csv
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO
from envs.env_1uav_5user_power import UAV6GMultiUserEnv

MODEL_PATH = "./models/uav_v5_6_sinr_rho01_mixed50_smooth_1m_seed42/model"
EVALUATION_SEED = 42
EVALUATION_RHO = 0.1
MAX_STEPS = 200

TEST_CONFIGS = [
    ("Evacuation + Default", "Original", "evacuation", "default"),
    ("Straggler + Default", "Original", "straggler", "default"),
    ("Random Walk + Default", "Unseen mobility", "random_walk", "default"),
    ("Straggler + Dense", "Unseen layout", "straggler", "dense"),
    ("Random Walk + Dense", "Combined shift", "random_walk", "dense"),
]

OUTPUT_DIR = f"./evaluations/generalisation/eval_seed{EVALUATION_SEED}"


def evaluate_configuration(model: PPO, scenario: str, building_layout: str) -> Dict[str, float]:
    env = UAV6GMultiUserEnv(
        num_users=5,
        mode="general",
        scenario=scenario,
        use_interference=True,
        interference_coupling=EVALUATION_RHO,
        power_allocation_mode="learned",
        building_layout=building_layout,
    )
    obs, _ = env.reset(seed=EVALUATION_SEED)

    histories = {k: [] for k in [
        "ee", "min_rate", "avg_rate", "jain", "qos", "los", "nlos"
    ]}

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        histories["ee"].append(float(info["energy_efficiency"]) / 1e6)
        histories["min_rate"].append(float(info["min_rate_mbps"]))
        histories["avg_rate"].append(float(info["avg_rate_mbps"]))
        histories["jain"].append(float(info["jain_index"]))
        histories["qos"].append(float(info["qos_violation_ratio"]))
        histories["los"].append(float(info["los_user_count"]))
        histories["nlos"].append(float(info["nlos_user_count"]))

        if terminated or truncated:
            break

    env.close()

    return {
        "mean_ee_mbit_j": float(np.mean(histories["ee"])),
        "mean_min_rate_mbps": float(np.mean(histories["min_rate"])),
        "mean_avg_rate_mbps": float(np.mean(histories["avg_rate"])),
        "mean_jain": float(np.mean(histories["jain"])),
        "mean_qos_violation": float(np.mean(histories["qos"])),
        "mean_los_users": float(np.mean(histories["los"])),
        "mean_nlos_users": float(np.mean(histories["nlos"])),
        "episode_steps": len(histories["ee"]),
    }


def run_generalisation_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}.zip")

    model = PPO.load(MODEL_PATH, device="cpu")
    results = []

    print("=" * 80)
    print("Generalisation Evaluation")
    print(f"Evaluation seed={EVALUATION_SEED}, rho={EVALUATION_RHO}, steps={MAX_STEPS}")
    print("=" * 80)

    for name, test_type, scenario, layout in TEST_CONFIGS:
        metrics = evaluate_configuration(model, scenario, layout)
        row = {
            "test_name": name,
            "test_type": test_type,
            "scenario": scenario,
            "building_layout": layout,
            **metrics,
        }
        results.append(row)
        print(
            f"{name}: EE={metrics['mean_ee_mbit_j']:.2f}, "
            f"MinRate={metrics['mean_min_rate_mbps']:.2f}, "
            f"Jain={metrics['mean_jain']:.4f}, "
            f"QoS={metrics['mean_qos_violation']:.4f}, "
            f"LoS={metrics['mean_los_users']:.2f}, "
            f"NLoS={metrics['mean_nlos_users']:.2f}"
        )

    csv_path = os.path.join(OUTPUT_DIR, "generalisation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    npy_path = os.path.join(OUTPUT_DIR, "generalisation_results.npy")
    np.save(npy_path, {
        "model_path": MODEL_PATH,
        "evaluation_seed": EVALUATION_SEED,
        "rho": EVALUATION_RHO,
        "max_steps": MAX_STEPS,
        "results": results,
    }, allow_pickle=True)

    print("\n" + "=" * 80)
    print("Generalisation Summary")
    print("=" * 80)
    print(f"{'Configuration':<25}{'EE':>10}{'MinRate':>12}{'Jain':>10}{'QoS':>10}")
    print("-" * 67)
    for r in results:
        print(
            f"{r['test_name']:<25}"
            f"{r['mean_ee_mbit_j']:>10.2f}"
            f"{r['mean_min_rate_mbps']:>12.2f}"
            f"{r['mean_jain']:>10.4f}"
            f"{r['mean_qos_violation']:>10.4f}"
        )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {npy_path}")


if __name__ == "__main__":
    run_generalisation_evaluation()