import csv
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from envs.env_1uav_5user_power import UAV6GMultiUserEnv


# ============================================================
# Experiment configuration
# ============================================================

GROUP_A_EXPERIMENT = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_smooth_1m_seed42"
)

SEED = 42

SCENARIOS = [
    "evacuation",
    "straggler",
]

INTERFERENCE_COUPLING = 0.1

OUTPUT_DIR = os.path.join(
    "./evaluations",
    "interference_allocation_2x2_seed42",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Loading
# ============================================================

def load_group_a_result(
    scenario: str,
) -> Dict:

    result_path = os.path.join(
        "./evaluations",
        GROUP_A_EXPERIMENT,
        scenario,
        f"seed{SEED}",
        f"eval_{scenario}_seed{SEED}_data.npy",
    )

    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Group A evaluation file not found:\n"
            f"{result_path}"
        )

    result = np.load(
        result_path,
        allow_pickle=True,
    ).item()

    return result


# ============================================================
# Communication calculation
# ============================================================

def calculate_rates(
    channel_gains: np.ndarray,
    transmit_power_per_step: np.ndarray,
    allocation_fractions: np.ndarray,
    bandwidth_hz: float,
    noise_power_w: float,
    use_interference: bool,
    interference_coupling: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate per-user power, interference, SINR/SNR and rate.

    Shapes:
        channel_gains:             [T, N]
        transmit_power_per_step:   [T, 1]
        allocation_fractions:      [T, N]
    """

    user_power_allocations = (
        transmit_power_per_step
        * allocation_fractions
    )

    desired_signal_powers = (
        user_power_allocations
        * channel_gains
    )

    if use_interference:

        other_user_powers = (
            transmit_power_per_step
            - user_power_allocations
        )

        interference_powers = (
            float(interference_coupling)
            * channel_gains
            * np.maximum(
                other_user_powers,
                0.0,
            )
        )

    else:

        interference_powers = np.zeros_like(
            desired_signal_powers
        )

    link_quality = (
        desired_signal_powers
        / (
            interference_powers
            + float(noise_power_w)
        )
    )

    link_quality = np.maximum(
        link_quality,
        1e-12,
    )

    rates_mbps = (
        float(bandwidth_hz)
        * np.log2(
            1.0 + link_quality
        )
        / 1e6
    )

    return (
        user_power_allocations,
        interference_powers,
        link_quality,
        rates_mbps,
    )


# ============================================================
# Metric calculation
# ============================================================

def calculate_jain(
    rates_mbps: np.ndarray,
) -> np.ndarray:

    rate_sums = np.sum(
        rates_mbps,
        axis=1,
    )

    rate_square_sums = np.sum(
        rates_mbps ** 2,
        axis=1,
    )

    num_users = rates_mbps.shape[1]

    return (
        rate_sums ** 2
        / (
            num_users
            * rate_square_sums
            + 1e-12
        )
    )


def summarize_condition(
    rates_mbps: np.ndarray,
    total_power_w: np.ndarray,
    qos_threshold_mbps: float,
) -> Dict[str, float]:

    per_step_min_rate = np.min(
        rates_mbps,
        axis=1,
    )

    per_step_avg_rate = np.mean(
        rates_mbps,
        axis=1,
    )

    per_step_sum_rate = np.sum(
        rates_mbps,
        axis=1,
    )

    per_step_jain = calculate_jain(
        rates_mbps
    )

    qos_violation_mask = (
        rates_mbps
        < float(qos_threshold_mbps)
    )

    # Mbps / W is numerically Mbit/J.
    per_step_ee = (
        per_step_sum_rate
        / np.maximum(
            total_power_w,
            1e-12,
        )
    )

    return {
        "mean_min_rate_mbps": float(
            np.mean(
                per_step_min_rate
            )
        ),

        "mean_avg_rate_mbps": float(
            np.mean(
                per_step_avg_rate
            )
        ),

        "mean_sum_rate_mbps": float(
            np.mean(
                per_step_sum_rate
            )
        ),

        "mean_jain": float(
            np.mean(
                per_step_jain
            )
        ),

        "qos_violation_ratio": float(
            np.mean(
                qos_violation_mask
            )
        ),

        "qos_violation_count": int(
            np.sum(
                qos_violation_mask
            )
        ),

        "mean_ee_mbit_j": float(
            np.mean(
                per_step_ee
            )
        ),

        "fifth_percentile_user_rate_mbps": float(
            np.percentile(
                rates_mbps,
                5,
            )
        ),
    }


# ============================================================
# Conditional rescue analysis
# ============================================================

def calculate_allocation_effect(
    equal_rates: np.ndarray,
    learned_rates: np.ndarray,
    learned_fractions: np.ndarray,
    qos_threshold_mbps: float,
) -> Dict[str, float]:

    num_steps = equal_rates.shape[0]
    num_users = equal_rates.shape[1]

    step_indices = np.arange(
        num_steps
    )

    equal_weakest_user_indices = np.argmin(
        equal_rates,
        axis=1,
    )

    weakest_equal_rates = equal_rates[
        step_indices,
        equal_weakest_user_indices,
    ]

    weakest_learned_rates = learned_rates[
        step_indices,
        equal_weakest_user_indices,
    ]

    weakest_rate_gain = (
        weakest_learned_rates
        - weakest_equal_rates
    )

    weakest_learned_fractions = (
        learned_fractions[
            step_indices,
            equal_weakest_user_indices,
        ]
    )

    equal_fraction_reference = (
        1.0 / num_users
    )

    weakest_allocation_advantage = (
        weakest_learned_fractions
        - equal_fraction_reference
    )

    equal_violation_mask = (
        equal_rates
        < float(qos_threshold_mbps)
    )

    learned_violation_mask = (
        learned_rates
        < float(qos_threshold_mbps)
    )

    rescued_mask = (
        equal_violation_mask
        & (~learned_violation_mask)
    )

    new_violation_mask = (
        (~equal_violation_mask)
        & learned_violation_mask
    )

    equal_violation_count = int(
        np.sum(
            equal_violation_mask
        )
    )

    rescued_count = int(
        np.sum(
            rescued_mask
        )
    )

    new_violation_count = int(
        np.sum(
            new_violation_mask
        )
    )

    if equal_violation_count > 0:
        rescue_rate = float(
            rescued_count
            / equal_violation_count
        )
    else:
        rescue_rate = np.nan

    tolerance = 1e-6

    return {
        "weakest_user_mean_rate_gain_mbps": float(
            np.mean(
                weakest_rate_gain
            )
        ),

        "weakest_user_mean_allocation_advantage": float(
            np.mean(
                weakest_allocation_advantage
            )
        ),

        "weakest_user_extra_power_step_ratio": float(
            np.mean(
                weakest_allocation_advantage
                > tolerance
            )
        ),

        "weakest_user_rate_gain_step_ratio": float(
            np.mean(
                weakest_rate_gain
                > tolerance
            )
        ),

        "qos_rescue_count": rescued_count,

        "qos_rescue_rate": rescue_rate,

        "new_qos_violation_count":
            new_violation_count,

        "net_qos_violation_reduction": int(
            rescued_count
            - new_violation_count
        ),
    }


# ============================================================
# Main scenario analysis
# ============================================================

def analyse_scenario(
    scenario: str,
    bandwidth_hz: float,
    noise_power_w: float,
) -> Dict:

    data = load_group_a_result(
        scenario
    )

    learned_fractions = np.asarray(
        data["power_allocation_fractions"],
        dtype=np.float64,
    )

    num_steps, num_users = (
        learned_fractions.shape
    )

    equal_fractions = np.full(
        (
            num_steps,
            num_users,
        ),
        1.0 / num_users,
        dtype=np.float64,
    )

    transmit_power = np.asarray(
        data["transmit_power"],
        dtype=np.float64,
    ).reshape(-1, 1)

    total_power = np.asarray(
        data["total_power"],
        dtype=np.float64,
    ).reshape(-1)

    path_losses_db = np.asarray(
        data["user_path_losses_db"],
        dtype=np.float64,
    )

    channel_gains = (
        10.0
        ** (
            -path_losses_db
            / 10.0
        )
    )

    qos_threshold = float(
        data["qos_threshold_mbps"]
    )

    # --------------------------------------------------------
    # E1: SNR + Equal allocation
    # --------------------------------------------------------

    (
        e1_power,
        e1_interference,
        e1_link_quality,
        e1_rates,
    ) = calculate_rates(
        channel_gains=channel_gains,
        transmit_power_per_step=transmit_power,
        allocation_fractions=equal_fractions,
        bandwidth_hz=bandwidth_hz,
        noise_power_w=noise_power_w,
        use_interference=False,
        interference_coupling=0.0,
    )

    # --------------------------------------------------------
    # E2: SNR + Learned allocation
    # --------------------------------------------------------

    (
        e2_power,
        e2_interference,
        e2_link_quality,
        e2_rates,
    ) = calculate_rates(
        channel_gains=channel_gains,
        transmit_power_per_step=transmit_power,
        allocation_fractions=learned_fractions,
        bandwidth_hz=bandwidth_hz,
        noise_power_w=noise_power_w,
        use_interference=False,
        interference_coupling=0.0,
    )

    # --------------------------------------------------------
    # E3: SINR + Equal allocation
    # --------------------------------------------------------

    (
        e3_power,
        e3_interference,
        e3_link_quality,
        e3_rates,
    ) = calculate_rates(
        channel_gains=channel_gains,
        transmit_power_per_step=transmit_power,
        allocation_fractions=equal_fractions,
        bandwidth_hz=bandwidth_hz,
        noise_power_w=noise_power_w,
        use_interference=True,
        interference_coupling=INTERFERENCE_COUPLING,
    )

    # --------------------------------------------------------
    # E4: SINR + Learned allocation
    # --------------------------------------------------------

    (
        e4_power,
        e4_interference,
        e4_link_quality,
        e4_rates,
    ) = calculate_rates(
        channel_gains=channel_gains,
        transmit_power_per_step=transmit_power,
        allocation_fractions=learned_fractions,
        bandwidth_hz=bandwidth_hz,
        noise_power_w=noise_power_w,
        use_interference=True,
        interference_coupling=INTERFERENCE_COUPLING,
    )

    condition_rates = {
        "E1_SNR_Equal": e1_rates,
        "E2_SNR_Learned": e2_rates,
        "E3_SINR_Equal": e3_rates,
        "E4_SINR_Learned": e4_rates,
    }

    summaries = {
        condition_name: summarize_condition(
            rates_mbps=rates,
            total_power_w=total_power,
            qos_threshold_mbps=qos_threshold,
        )
        for condition_name, rates
        in condition_rates.items()
    }

    allocation_effect_snr = (
        calculate_allocation_effect(
            equal_rates=e1_rates,
            learned_rates=e2_rates,
            learned_fractions=learned_fractions,
            qos_threshold_mbps=qos_threshold,
        )
    )

    allocation_effect_sinr = (
        calculate_allocation_effect(
            equal_rates=e3_rates,
            learned_rates=e4_rates,
            learned_fractions=learned_fractions,
            qos_threshold_mbps=qos_threshold,
        )
    )

    # --------------------------------------------------------
    # Interaction effects
    # --------------------------------------------------------

    snr_min_rate_gain = (
        summaries["E2_SNR_Learned"][
            "mean_min_rate_mbps"
        ]
        - summaries["E1_SNR_Equal"][
            "mean_min_rate_mbps"
        ]
    )

    sinr_min_rate_gain = (
        summaries["E4_SINR_Learned"][
            "mean_min_rate_mbps"
        ]
        - summaries["E3_SINR_Equal"][
            "mean_min_rate_mbps"
        ]
    )

    min_rate_interaction_effect = (
        sinr_min_rate_gain
        - snr_min_rate_gain
    )

    snr_qos_reduction = (
        summaries["E1_SNR_Equal"][
            "qos_violation_ratio"
        ]
        - summaries["E2_SNR_Learned"][
            "qos_violation_ratio"
        ]
    )

    sinr_qos_reduction = (
        summaries["E3_SINR_Equal"][
            "qos_violation_ratio"
        ]
        - summaries["E4_SINR_Learned"][
            "qos_violation_ratio"
        ]
    )

    qos_interaction_effect = (
        sinr_qos_reduction
        - snr_qos_reduction
    )

    snr_jain_gain = (
        summaries["E2_SNR_Learned"][
            "mean_jain"
        ]
        - summaries["E1_SNR_Equal"][
            "mean_jain"
        ]
    )

    sinr_jain_gain = (
        summaries["E4_SINR_Learned"][
            "mean_jain"
        ]
        - summaries["E3_SINR_Equal"][
            "mean_jain"
        ]
    )

    jain_interaction_effect = (
        sinr_jain_gain
        - snr_jain_gain
    )

    # --------------------------------------------------------
    # Validation against Group A recorded SINR rates
    # --------------------------------------------------------

    recorded_group_a_rates = np.asarray(
        data["rates"],
        dtype=np.float64,
    )

    max_rate_reconstruction_error = float(
        np.max(
            np.abs(
                e4_rates
                - recorded_group_a_rates
            )
        )
    )

    interaction_summary = {
        "snr_min_rate_allocation_gain_mbps":
            float(snr_min_rate_gain),

        "sinr_min_rate_allocation_gain_mbps":
            float(sinr_min_rate_gain),

        "min_rate_interaction_effect_mbps":
            float(min_rate_interaction_effect),

        "snr_qos_reduction":
            float(snr_qos_reduction),

        "sinr_qos_reduction":
            float(sinr_qos_reduction),

        "qos_interaction_effect":
            float(qos_interaction_effect),

        "snr_jain_gain":
            float(snr_jain_gain),

        "sinr_jain_gain":
            float(sinr_jain_gain),

        "jain_interaction_effect":
            float(jain_interaction_effect),

        "max_rate_reconstruction_error_mbps":
            max_rate_reconstruction_error,
    }

    return {
        "scenario": scenario,
        "qos_threshold_mbps": qos_threshold,
        "summaries": summaries,
        "allocation_effect_snr":
            allocation_effect_snr,
        "allocation_effect_sinr":
            allocation_effect_sinr,
        "interaction_summary":
            interaction_summary,

        "rates": condition_rates,

        "learned_fractions":
            learned_fractions,

        "equal_fractions":
            equal_fractions,

        "channel_gains":
            channel_gains,

        "transmit_power":
            transmit_power,

        "total_power":
            total_power,

        "link_quality": {
            "E1_SNR_Equal":
                e1_link_quality,
            "E2_SNR_Learned":
                e2_link_quality,
            "E3_SINR_Equal":
                e3_link_quality,
            "E4_SINR_Learned":
                e4_link_quality,
        },

        "interference_powers": {
            "E1_SNR_Equal":
                e1_interference,
            "E2_SNR_Learned":
                e2_interference,
            "E3_SINR_Equal":
                e3_interference,
            "E4_SINR_Learned":
                e4_interference,
        },

        "power_allocations": {
            "E1_SNR_Equal":
                e1_power,
            "E2_SNR_Learned":
                e2_power,
            "E3_SINR_Equal":
                e3_power,
            "E4_SINR_Learned":
                e4_power,
        },
    }


# ============================================================
# Console table
# ============================================================

def print_scenario_table(
    result: Dict,
) -> None:

    scenario = result["scenario"]
    summaries = result["summaries"]

    condition_order = [
        "E1_SNR_Equal",
        "E2_SNR_Learned",
        "E3_SINR_Equal",
        "E4_SINR_Learned",
    ]

    metric_order = [
        (
            "mean_min_rate_mbps",
            "Mean minimum rate (Mbps)",
        ),
        (
            "mean_avg_rate_mbps",
            "Mean average rate (Mbps)",
        ),
        (
            "mean_sum_rate_mbps",
            "Mean sum rate (Mbps)",
        ),
        (
            "fifth_percentile_user_rate_mbps",
            "5th-percentile rate (Mbps)",
        ),
        (
            "qos_violation_ratio",
            "QoS violation ratio",
        ),
        (
            "qos_violation_count",
            "QoS violation count",
        ),
        (
            "mean_jain",
            "Mean Jain index",
        ),
        (
            "mean_ee_mbit_j",
            "Mean EE (Mbit/J)",
        ),
    ]

    print("\n" + "=" * 125)

    print(
        f"2×2 Interference–Allocation Test — "
        f"{scenario.capitalize()}"
    )

    print("=" * 125)

    print(
        f"{'Metric':<36}"
        f"{'E1 SNR Equal':>20}"
        f"{'E2 SNR Learned':>22}"
        f"{'E3 SINR Equal':>21}"
        f"{'E4 SINR Learned':>23}"
    )

    print("-" * 125)

    for metric_key, metric_label in metric_order:

        values = [
            summaries[condition][metric_key]
            for condition in condition_order
        ]

        print(
            f"{metric_label:<36}"
            f"{values[0]:>20.4f}"
            f"{values[1]:>22.4f}"
            f"{values[2]:>21.4f}"
            f"{values[3]:>23.4f}"
        )

    interaction = result[
        "interaction_summary"
    ]

    print("\nInteraction effects")

    print("-" * 70)

    for key, value in interaction.items():

        print(
            f"{key:<48}"
            f"{value:>18.6f}"
        )

    print("\nAllocation behaviour under SNR")

    print("-" * 70)

    for key, value in result[
        "allocation_effect_snr"
    ].items():

        print(
            f"{key:<48}"
            f"{value:>18.6f}"
        )

    print("\nAllocation behaviour under SINR")

    print("-" * 70)

    for key, value in result[
        "allocation_effect_sinr"
    ].items():

        print(
            f"{key:<48}"
            f"{value:>18.6f}"
        )


# ============================================================
# CSV output
# ============================================================

def save_csv(
    result: Dict,
) -> str:

    scenario = result["scenario"]

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"{scenario}_2x2_metrics.csv",
    )

    summaries = result["summaries"]

    condition_order = [
        "E1_SNR_Equal",
        "E2_SNR_Learned",
        "E3_SINR_Equal",
        "E4_SINR_Learned",
    ]

    metric_keys = list(
        summaries[
            condition_order[0]
        ].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "Metric",
                *condition_order,
            ]
        )

        for metric_key in metric_keys:

            writer.writerow(
                [
                    metric_key,
                    *[
                        summaries[
                            condition
                        ][metric_key]
                        for condition
                        in condition_order
                    ],
                ]
            )

        writer.writerow([])

        writer.writerow(
            [
                "Interaction metric",
                "Value",
            ]
        )

        for key, value in result[
            "interaction_summary"
        ].items():

            writer.writerow(
                [
                    key,
                    value,
                ]
            )

    return csv_path


# ============================================================
# Comparison figure
# ============================================================

def save_figure(
    all_results: Dict[str, Dict],
) -> str:

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

    condition_order = [
        "E1_SNR_Equal",
        "E2_SNR_Learned",
        "E3_SINR_Equal",
        "E4_SINR_Learned",
    ]

    condition_labels = [
        "SNR + Equal",
        "SNR + Learned",
        "SINR + Equal",
        "SINR + Learned",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    x_positions = np.arange(
        len(SCENARIOS)
    )

    bar_width = 0.19

    offsets = np.array(
        [
            -1.5,
            -0.5,
            0.5,
            1.5,
        ]
    ) * bar_width

    for ax, (
        metric_key,
        title,
        unit,
    ) in zip(
        axes,
        metric_specs,
    ):

        for condition_index, condition in enumerate(
            condition_order
        ):

            values = [
                all_results[
                    scenario
                ]["summaries"][
                    condition
                ][metric_key]
                for scenario in SCENARIOS
            ]

            bars = ax.bar(
                x_positions
                + offsets[
                    condition_index
                ],
                values,
                width=bar_width,
                label=condition_labels[
                    condition_index
                ],
            )

            for bar in bars:

                height = float(
                    bar.get_height()
                )

                ax.annotate(
                    f"{height:.3f}",
                    xy=(
                        bar.get_x()
                        + bar.get_width() / 2,
                        height,
                    ),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(
            x_positions
        )

        ax.set_xticklabels(
            [
                scenario.capitalize()
                for scenario in SCENARIOS
            ]
        )

        ax.set_title(
            title
        )

        ax.set_ylabel(
            unit
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        "2×2 Interference–Allocation Factorial Test",
        fontsize=15,
    )

    fig.tight_layout()

    figure_path = os.path.join(
        OUTPUT_DIR,
        "interference_allocation_2x2.png",
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return figure_path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Instantiate only to read the same communication constants
    # used by the environment.
    reference_env = UAV6GMultiUserEnv(
        num_users=5,
        mode="general",
        scenario="static",
        use_interference=True,
        interference_coupling=(
            INTERFERENCE_COUPLING
        ),
        power_allocation_mode="learned",
    )

    bandwidth_hz = float(
        reference_env.B
    )

    noise_power_w = float(
        reference_env.sigma2
    )

    reference_env.close()

    print("\nEnvironment constants")
    print("=" * 70)
    print(
        f"Bandwidth: "
        f"{bandwidth_hz:.6e} Hz"
    )
    print(
        f"Noise power: "
        f"{noise_power_w:.6e} W"
    )
    print(
        f"Interference coupling: "
        f"{INTERFERENCE_COUPLING:.4f}"
    )

    all_results = {}

    generated_csv_files = []

    for scenario in SCENARIOS:

        scenario_result = analyse_scenario(
            scenario=scenario,
            bandwidth_hz=bandwidth_hz,
            noise_power_w=noise_power_w,
        )

        all_results[
            scenario
        ] = scenario_result

        print_scenario_table(
            scenario_result
        )

        generated_csv_files.append(
            save_csv(
                scenario_result
            )
        )

    result_path = os.path.join(
        OUTPUT_DIR,
        "interference_allocation_2x2_data.npy",
    )

    np.save(
        result_path,
        all_results,
        allow_pickle=True,
    )

    figure_path = save_figure(
        all_results
    )

    print("\nGenerated files")
    print("=" * 70)

    for csv_path in generated_csv_files:
        print(
            f"- {csv_path}"
        )

    print(
        f"- {result_path}"
    )

    print(
        f"- {figure_path}"
    )