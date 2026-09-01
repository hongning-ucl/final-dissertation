import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


GROUP_A_EXPERIMENT = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_smooth_1m_seed42"
)

GROUP_C_EXPERIMENT = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_equalalloc_1m_seed42"
)

SEED = 42

SCENARIOS = [
    "evacuation",
    "straggler",
]


def load_evaluation(
    experiment_name: str,
    scenario: str,
    seed: int,
) -> Dict:
    path = os.path.join(
        "./evaluations",
        experiment_name,
        scenario,
        f"seed{seed}",
        f"eval_{scenario}_seed{seed}_data.npy",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=True,
    ).item()

    return data


def summarize(data: Dict) -> Dict[str, float]:
    return {
        "mean_reward": float(
            np.mean(data["reward"])
        ),
        "mean_ee_mbit_j": float(
            np.mean(data["ee"]) / 1e6
        ),
        "mean_total_power_w": float(
            np.mean(data["total_power"])
        ),
        "mean_min_rate_mbps": float(
            np.mean(data["min_rate"])
        ),
        "mean_avg_rate_mbps": float(
            np.mean(data["avg_rate"])
        ),
        "mean_jain": float(
            np.mean(data["jain"])
        ),
        "qos_violation_ratio": float(
            np.mean(
                data["qos_violation_ratio"]
            )
        ),
        "mean_tracking_distance_m": float(
            np.mean(
                data["tracking_distance"]
            )
        ),
        "travel_distance_m": float(
            data["total_uav_travel_distance"]
        ),
    }


all_results = {}

for scenario in SCENARIOS:
    group_a_data = load_evaluation(
        GROUP_A_EXPERIMENT,
        scenario,
        SEED,
    )

    group_c_data = load_evaluation(
        GROUP_C_EXPERIMENT,
        scenario,
        SEED,
    )

    group_a_summary = summarize(
        group_a_data
    )

    group_c_summary = summarize(
        group_c_data
    )

    all_results[scenario] = {
        "Group A": group_a_summary,
        "Group C": group_c_summary,
    }

    print("\n" + "=" * 90)
    print(
        f"Group A vs Group C — "
        f"{scenario.capitalize()}"
    )
    print("=" * 90)

    print(
        f"{'Metric':<32}"
        f"{'Group A':>16}"
        f"{'Group C':>16}"
        f"{'A - C':>16}"
    )

    print("-" * 90)

    for metric_name in group_a_summary:
        a_value = group_a_summary[
            metric_name
        ]
        c_value = group_c_summary[
            metric_name
        ]

        difference = (
            a_value - c_value
        )

        print(
            f"{metric_name:<32}"
            f"{a_value:>16.4f}"
            f"{c_value:>16.4f}"
            f"{difference:>16.4f}"
        )


# ============================================================
# Grouped comparison figure
# ============================================================

metric_specs = [
    (
        "mean_min_rate_mbps",
        "Mean Minimum Rate",
        "Mbps",
    ),
    (
        "qos_violation_ratio",
        "QoS Violation Ratio",
        "Ratio",
    ),
    (
        "mean_jain",
        "Jain Fairness Index",
        "Index",
    ),
    (
        "mean_ee_mbit_j",
        "Energy Efficiency",
        "Mbit/J",
    ),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
)

axes = axes.flatten()

x = np.arange(
    len(SCENARIOS)
)

bar_width = 0.35

for ax, (
    metric_key,
    metric_title,
    metric_unit,
) in zip(
    axes,
    metric_specs,
):
    group_a_values = [
        all_results[scenario][
            "Group A"
        ][metric_key]
        for scenario in SCENARIOS
    ]

    group_c_values = [
        all_results[scenario][
            "Group C"
        ][metric_key]
        for scenario in SCENARIOS
    ]

    ax.bar(
        x - bar_width / 2,
        group_a_values,
        width=bar_width,
        label="Group A: Full PPO",
    )

    ax.bar(
        x + bar_width / 2,
        group_c_values,
        width=bar_width,
        label="Group C: Equal allocation PPO",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            scenario.capitalize()
            for scenario in SCENARIOS
        ]
    )

    ax.set_title(metric_title)
    ax.set_ylabel(metric_unit)
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    ax.legend(fontsize=8)

fig.suptitle(
    "Group A vs Group C: "
    "Effect of Learned Per-User Power Allocation",
    fontsize=14,
)

fig.tight_layout()

output_path = (
    "./evaluations/"
    "group_a_vs_c_seed42.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

np.save(
    "./evaluations/"
    "group_a_vs_c_seed42.npy",
    all_results,
    allow_pickle=True,
)

print("\nSaved:")
print(f"- {output_path}")
print(
    "- ./evaluations/"
    "group_a_vs_c_seed42.npy"
)