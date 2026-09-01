import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import PPO

from envs.env_1uav_5user_power import UAV6GMultiUserEnv


# Existing checkpoint used for code validation.
# Keep its original V5.3 filename because it was trained previously.

# Group A: PPO trained with smooth power allocation
EXPERIMENT_NAME = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_smooth_1m_seed42"
)
POWER_ALLOCATION_MODE = "learned"
MODEL_GROUP = "A_full_ppo"
# TRAINING_MOBILITY = "mixed"
# TEST_SEED = 42

# Group C:
# EXPERIMENT_NAME = (
#     "uav_v5_6_sinr_rho01_"
#     "mixed50_equalalloc_1m_seed42"
# )
# POWER_ALLOCATION_MODE = "equal"
# MODEL_GROUP = "C_equal_allocation_ppo"

# TRAINING_MOBILITY = "mixed"

# TEST_SEED = 42

# Group D: PPO trained with SNR model, tested under SINR
# EXPERIMENT_NAME = (
#     "uav_v5_6_snr_"
#     "mixed50_smooth_1m_seed42"
# )
# POWER_ALLOCATION_MODE = "learned"
# MODEL_GROUP = (
#     "D_snr_trained_ppo_tested_under_sinr"
# )
TRAINING_MOBILITY = "mixed"
TEST_SEED = 42

MODEL_PATH = (
    f"./models/{EXPERIMENT_NAME}/model"
)
TRAINING_CHANNEL_MODEL = "sinr"
EVALUATION_CHANNEL_MODEL = "sinr"

# ============================================================
# Rho sensitivity experiment
# ============================================================

RHO_VALUES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

GROUP_A_EXPERIMENT = (
    "uav_v5_6_sinr_rho01_"
    "mixed50_smooth_1m_seed42"
)

GROUP_D_EXPERIMENT = (
    "uav_v5_6_snr_"
    "mixed50_smooth_1m_seed42"
)

GROUP_A_MODEL_PATH = (
    f"./models/{GROUP_A_EXPERIMENT}/model"
)

GROUP_D_MODEL_PATH = (
    f"./models/{GROUP_D_EXPERIMENT}/model"
)

def _building_vertices(building: Dict[str, float]) -> List[List[Tuple[float, float, float]]]:
    x0, x1 = building["x_min"], building["x_max"]
    y0, y1 = building["y_min"], building["y_max"]
    z0, z1 = 0.0, building["height"]

    p000 = (x0, y0, z0)
    p100 = (x1, y0, z0)
    p110 = (x1, y1, z0)
    p010 = (x0, y1, z0)
    p001 = (x0, y0, z1)
    p101 = (x1, y0, z1)
    p111 = (x1, y1, z1)
    p011 = (x0, y1, z1)

    return [
        [p000, p100, p110, p010],
        [p001, p101, p111, p011],
        [p000, p100, p101, p001],
        [p100, p110, p111, p101],
        [p110, p010, p011, p111],
        [p010, p000, p001, p011],
    ]


def _project_points(points_xy: np.ndarray, origin_xy: np.ndarray, direction_xy: np.ndarray) -> np.ndarray:
    return (points_xy - origin_xy[None, :]) @ direction_xy


def _project_building_interval(
    building: Dict[str, float],
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> Tuple[float, float]:
    corners = np.array(
        [
            [building["x_min"], building["y_min"]],
            [building["x_min"], building["y_max"]],
            [building["x_max"], building["y_min"]],
            [building["x_max"], building["y_max"]],
        ],
        dtype=np.float64,
    )
    projected = _project_points(corners, origin_xy, direction_xy)
    return float(projected.min()), float(projected.max())

def moving_average(
    values: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Return a centred moving average with edge padding."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if window <= 1:
        return values.copy()

    kernel = np.ones(
        window,
        dtype=np.float64,
    ) / window

    left_padding = window // 2
    right_padding = (
        window - 1 - left_padding
    )

    padded = np.pad(
        values,
        (left_padding, right_padding),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def _run_baseline(
    mode: str,
    scenario: str,
    use_interference: bool,
    interference_coupling: float,
    power_allocation_mode: str,
    seed: int,
    max_steps: int,
    centroid: np.ndarray,
    building_layout:str = "default",
) -> Dict[str, np.ndarray]:
    # env = UAV6GMultiUserEnv(num_users=5, mode=mode, scenario=scenario,    use_interference=use_interference,
    # interference_coupling=interference_coupling,)
    env = UAV6GMultiUserEnv(
        num_users=5,
        mode=mode,
        scenario=scenario,
        use_interference=use_interference,
        interference_coupling=interference_coupling,
        power_allocation_mode=power_allocation_mode,
        building_layout=building_layout,
    )
    env.reset(seed=seed)

    target_pos = np.array([centroid[0], centroid[1], 30.0], dtype=np.float32)
    env.unwrapped.uav_pos = target_pos.copy()
    env.unwrapped.last_pos = target_pos.copy()

    baseline_centroid = np.mean(env.unwrapped.user_positions, axis=0)
    env.unwrapped.previous_tracking_distance = float(
        np.linalg.norm(target_pos[:2] - baseline_centroid)
    )

    action = np.zeros(env.action_space.shape, dtype=np.float32)

    history = {
        "reward": [],
        "ee": [],
        "power": [],
        "avg_rate": [],
        "min_rate": [],

    }

    for _ in range(max_steps):
        _, reward, terminated, truncated, info = env.step(action)
        history["reward"].append(float(reward))
        history["ee"].append(float(info["energy_efficiency"]))
        history["power"].append(float(info["total_power"]))
        history["avg_rate"].append(float(info["avg_rate_mbps"]))
        history["min_rate"].append(float(info["min_rate_mbps"]))
        if terminated or truncated:
            break

    env.close()
    return {key: np.asarray(value) for key, value in history.items()}


def evaluate_and_visualize(
    num_users=5,
    mode: str="general",
    scenario: str="evacuation",
    use_interference: bool=False,
    interference_coupling: float=0.0,
    seed: int=42,
    max_steps: int=200,
    building_layout: str="default",
):
    scenario = scenario.lower().strip()

    supported_scenarios = {
        "static",
        "evacuation",
        "straggler",
        "random_walk",
    }

    if scenario not in supported_scenarios:
        raise ValueError(
            f"Unsupported evaluation scenario: "
            f"{scenario}. Supported scenarios are: "
            f"{sorted(supported_scenarios)}"
        )

    if not 0.0 <= interference_coupling <= 1.0:
        raise ValueError(
            "interference_coupling must be "
            "between 0 and 1."
        )

    if not use_interference and interference_coupling != 0.0:
        raise ValueError(
            "For SNR evaluation, set "
            "interference_coupling=0.0."
        )
    channel_model = (
        "sinr"
        if use_interference
        else "snr"
    )

    evaluation_dir = os.path.join(
        "./evaluations",
        EXPERIMENT_NAME,
        scenario,
        building_layout,
        f"seed{seed}"
    )

    os.makedirs(
        evaluation_dir,
        exist_ok=True
    )

    output_prefix = os.path.join(
        evaluation_dir,
        f"eval_{scenario}_seed{seed}"
    )
    env = UAV6GMultiUserEnv(
        num_users=5,
        mode=mode,
        scenario=scenario,
        use_interference=use_interference,
        interference_coupling=interference_coupling,
        power_allocation_mode=POWER_ALLOCATION_MODE,
        building_layout=building_layout,
    )

    print("\n" + "=" * 70)
    print("Evaluation configuration")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Mode: {mode}")
    print(f"Scenario: {scenario}")
    print(f"Seed: {seed}")
    print(f"Maximum steps: {max_steps}")
    print(f"Output prefix: {output_prefix}")
    print(f"Channel model: {channel_model}")
    print(f"Use interference: {use_interference}")
    print(
        f"Interference coupling: "
        f"{interference_coupling}"
    )
    print(
        f"Model group: "
        f"{MODEL_GROUP}"
    )

    print(
        f"Power allocation mode: "
        f"{POWER_ALLOCATION_MODE}"
    )
    print("=" * 70)

    if not os.path.exists(MODEL_PATH) and not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}.zip")

    model_name_lower = MODEL_PATH.lower()

    # Group D intentionally evaluates an SNR-trained model
    # under a SINR environment.
    is_group_d = (
        MODEL_GROUP
        == "D_snr_trained_ppo_tested_under_sinr"
    )

    if (
        "snr" in model_name_lower
        and use_interference
        and not is_group_d
    ):
        raise ValueError(
            "The selected model appears to be an SNR model, "
            "but use_interference=True. "
            "This is only allowed for Group D."
        )

    if (
        "sinr" in model_name_lower
        and not use_interference
    ):
        raise ValueError(
            "The selected model appears to be a SINR model, "
            "but use_interference=False."
        )

    model = PPO.load(MODEL_PATH,
    device="cpu",)
    obs, _ = env.reset(seed=seed)

    initial_users = env.unwrapped.user_positions.copy()
    initial_centroid = np.mean(initial_users, axis=0)
    raw_target = env.unwrapped.evacuation_target

    evacuation_target = (
        None
        if raw_target is None
        else raw_target.copy()
    )

    baseline = _run_baseline(
        mode=mode,
        scenario=scenario,
        use_interference=use_interference,
        interference_coupling=interference_coupling,
        power_allocation_mode=POWER_ALLOCATION_MODE,
        seed=seed,
        max_steps=max_steps,
        centroid=initial_centroid,
        building_layout=building_layout,
    )

    obs, _ = env.reset(seed=seed)

    uav_history = [env.unwrapped.uav_pos.copy()]
    user_positions_history = [env.unwrapped.user_positions.copy()]
    centroid_history = [np.mean(env.unwrapped.user_positions, axis=0).copy()]

    histories: Dict[str, List] = {
        "reward": [],
        "ee": [],
        "total_power": [],
        "aero_power": [],
        "transmit_power": [],
        "electronic_power": [],
        "avg_rate": [],
        "min_rate": [],
        "jain": [],
        "tracking_distance": [],
        "mean_user_distance": [],
        "max_user_distance": [],
        "rates": [],
        "los_flags": [],
        "power_allocation": [],
        "user_path_losses_db": [],
        "user_distances": [],
        "qos_violation_ratio": [],
        "user_sinrs": [],
        "user_interference_powers": [],
        "power_allocation_fractions": [],
        "altitude": [],
        "boundary_violation": [],
        "building_collision": [],
        "altitude_violation": [],
        "reward_ee": [],
        "reward_mean_qos": [],
        "reward_worst_qos": [],
        "reward_min_rate": [],
        "reward_tracking_distance": [],
        "reward_tracking_progress": [],
        "reward_movement": [],
        "power_allocation_change": [],
        "reward_power_allocation_change": [],
        "raw_power_weights": [],
        "smoothed_power_weights": [],
        "user_velocities": [],
        "altitude_hold_active": [],
        "altitude_hold_target": [],
        "altitude_hold_error": [],
        "vertical_action_raw": [],
        "vertical_action_applied": [],
    }

    qos_threshold_mbps = None

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if qos_threshold_mbps is None:
            qos_threshold_mbps = float(info["qos_threshold_mbps"])

        uav_history.append(env.unwrapped.uav_pos.copy())
        user_positions_history.append(np.asarray(info["user_positions"]).copy())
        centroid_history.append(np.asarray(info["user_centroid"]).copy())

        user_distances = np.asarray(info["user_distances"], dtype=np.float64)

        histories["reward"].append(float(reward))
        histories["ee"].append(float(info["energy_efficiency"]))
        histories["total_power"].append(float(info["total_power"]))
        histories["aero_power"].append(float(info["aerodynamic_power"]))
        histories["transmit_power"].append(float(info["transmit_power"]))
        histories["electronic_power"].append(float(info["electronic_power"]))
        histories["avg_rate"].append(float(info["avg_rate_mbps"]))
        histories["min_rate"].append(float(info["min_rate_mbps"]))
        histories["jain"].append(float(info["jain_index"]))
        histories["tracking_distance"].append(float(info["tracking_distance"]))
        histories["mean_user_distance"].append(float(np.mean(user_distances)))
        histories["max_user_distance"].append(float(np.max(user_distances)))
        histories["rates"].append(np.asarray(info["user_rates_mbps"]).copy())
        histories["los_flags"].append(np.asarray(info["los_flags"], dtype=bool).copy())
        histories["power_allocation"].append(
            np.asarray(
                info["power_allocation"],
                dtype=np.float64
            ).copy()
        )

        histories["power_allocation_fractions"].append(
            np.asarray(
                info["power_allocation_fractions"],
                dtype=np.float64
            ).copy()
        )

        histories["user_path_losses_db"].append(
            np.asarray(
                info["user_path_losses_db"],
                dtype=np.float64
            ).copy()
        )

        histories["user_distances"].append(
            user_distances.copy()
        )

        histories["user_sinrs"].append(
            np.asarray(
                info["user_sinrs"],
                dtype=np.float64
            ).copy()
        )

        histories["user_interference_powers"].append(
            np.asarray(
                info["user_interference_powers"],
                dtype=np.float64
            ).copy()
        )
        histories["qos_violation_ratio"].append(float(info["qos_violation_ratio"]))
        histories["altitude"].append(float(info["altitude"]))
        histories["boundary_violation"].append(bool(info["boundary_violation"]))
        histories["building_collision"].append(bool(info["building_collision"]))
        histories["altitude_violation"].append(bool(info["altitude_violation"]))
        histories["reward_ee"].append(float(info.get("reward_ee_component", 0.0)))
        histories["reward_mean_qos"].append(float(info.get("reward_mean_qos_component", 0.0)))
        histories["reward_worst_qos"].append(float(info.get("reward_worst_user_qos_component", 0.0)))
        histories["reward_min_rate"].append(float(info.get("reward_min_rate_component", 0.0)))
        histories["reward_tracking_distance"].append(float(info.get("reward_tracking_distance_component", 0.0)))
        histories["reward_tracking_progress"].append(float(info.get("reward_tracking_progress_component", 0.0)))
        histories["reward_movement"].append(float(info.get("reward_movement_component", 0.0)))
        histories["power_allocation_change"].append(
            float(
                info.get(
                    "power_allocation_change_penalty",
                    0.0
                )
            )
        )

        histories[
            "reward_power_allocation_change"
        ].append(
            float(
                info.get(
                    "reward_power_allocation_change_component",
                    0.0
                )
            )
        )

        histories["raw_power_weights"].append(
            np.asarray(
                info.get(
                    "raw_power_weights",
                    np.zeros(env.num_users)
                )
            ).copy()
        )

        histories["smoothed_power_weights"].append(
            np.asarray(
                info.get(
                    "smoothed_power_weights",
                    np.zeros(env.num_users)
                )
            ).copy()
        )
        histories["user_velocities"].append(
            np.asarray(
                info["user_velocities"],
                dtype=np.float32
            ).copy()
        )
        # ============================================================
        # Altitude-hold diagnostics
        # ============================================================

        histories["altitude_hold_active"].append(
            bool(
                info.get(
                    "altitude_hold_active",
                    False
                )
            )
        )

        histories["altitude_hold_target"].append(
            float(
                info.get(
                    "altitude_hold_target",
                    np.nan
                )
            )
        )

        histories["altitude_hold_error"].append(
            float(
                info.get(
                    "altitude_hold_error",
                    0.0
                )
            )
        )

        histories["vertical_action_raw"].append(
            float(
                info.get(
                    "vertical_action_raw",
                    0.0
                )
            )
        )

        histories["vertical_action_applied"].append(
            float(
                info.get(
                    "vertical_action_applied",
                    0.0
                )
            )
        )
       
        if step_idx % 20 == 0 or terminated or truncated:
            print(
                f"step={step_idx + 1:03d} | reward={reward:7.4f} | "
                f"min_rate={info['min_rate_mbps']:.2f} Mbps | "
                f"EE={info['energy_efficiency'] / 1e6:.2f} Mbit/J | "
                f"tracking={info['tracking_distance']:.2f} m"
            )
            print(
                "User positions:\n",
                np.round(
                    np.asarray(info["user_positions"]),
                    2
                )
            )

            print(
                "User speeds:",
                np.round(
                    np.linalg.norm(
                        np.asarray(info["user_velocities"]),
                        axis=1
                    ),
                    3
                )
            )

            print(
                "Stationary users:",
                info["user_stationary_count"],
                "| Blocked users:",
                info["user_blocked_count"]
            )

        if terminated or truncated:
            break



    uav_history = np.asarray(uav_history, dtype=np.float32)
    user_positions_history = np.asarray(user_positions_history, dtype=np.float32)
    centroid_history = np.asarray(centroid_history, dtype=np.float32)
    data = {key: np.asarray(value) for key, value in histories.items()}
    # ============================================================
    # Same-state equal-allocation counterfactual
    # ============================================================

    # 当前 PPO 实际使用的 allocation fraction
    learned_allocation_fractions = np.asarray(
        data["power_allocation_fractions"],
        dtype=np.float64
    )

    num_evaluation_steps = learned_allocation_fractions.shape[0]
    num_evaluation_users = learned_allocation_fractions.shape[1]

    # Equal allocation: five users each receive 1/5
    equal_allocation_fractions = np.full(
        (
            num_evaluation_steps,
            num_evaluation_users
        ),
        1.0 / num_evaluation_users,
        dtype=np.float64
    )

    # 每一步的总发射功率
    transmit_power_per_step = np.asarray(
        data["transmit_power"],
        dtype=np.float64
    ).reshape(-1, 1)

    # Equal allocation 对应的用户功率
    equal_power_allocation = (
        transmit_power_per_step
        * equal_allocation_fractions
    )

    # 根据相同 timestep 的 path loss 恢复 channel gain
    channel_gains = (
        10.0
        ** (
            -np.asarray(
                data["user_path_losses_db"],
                dtype=np.float64
            )
            / 10.0
        )
    )

    equal_desired_signal_powers = (
        equal_power_allocation
        * channel_gains
    )

    equal_other_user_powers = (
        transmit_power_per_step
        - equal_power_allocation
    )

    if use_interference:
        equal_interference_powers = (
            float(interference_coupling)
            * channel_gains
            * np.maximum(
                equal_other_user_powers,
                0.0
            )
        )
    else:
        equal_interference_powers = np.zeros_like(
            equal_desired_signal_powers
        )

    equal_user_sinrs = (
        equal_desired_signal_powers
        / (
            equal_interference_powers
            + float(env.sigma2)
        )
    )

    equal_user_sinrs = np.maximum(
        equal_user_sinrs,
        1e-12
    )

    equal_user_rates_mbps = (
        float(env.B)
        * np.log2(
            1.0 + equal_user_sinrs
        )
        / 1e6
    )

    # ------------------------------------------------------------
    # Learned allocation 与 equal allocation 的核心指标
    # ------------------------------------------------------------

    learned_user_rates_mbps = np.asarray(
        data["rates"],
        dtype=np.float64
    )

    learned_min_rates_mbps = np.min(
        learned_user_rates_mbps,
        axis=1
    )

    equal_min_rates_mbps = np.min(
        equal_user_rates_mbps,
        axis=1
    )

    min_rate_improvement_mbps = (
        learned_min_rates_mbps
        - equal_min_rates_mbps
    )

    learned_sum_rates_mbps = np.sum(
        learned_user_rates_mbps,
        axis=1
    )

    equal_sum_rates_mbps = np.sum(
        equal_user_rates_mbps,
        axis=1
    )

    # Jain fairness
    learned_rate_square_sums = np.sum(
        learned_user_rates_mbps ** 2,
        axis=1
    )

    equal_rate_square_sums = np.sum(
        equal_user_rates_mbps ** 2,
        axis=1
    )

    learned_jain_counterfactual = (
        learned_sum_rates_mbps ** 2
        / (
            num_evaluation_users
            * learned_rate_square_sums
            + 1e-12
        )
    )

    equal_jain_counterfactual = (
        equal_sum_rates_mbps ** 2
        / (
            num_evaluation_users
            * equal_rate_square_sums
            + 1e-12
        )
    )

    # QoS violation ratio
    learned_qos_violation_counterfactual = np.mean(
        learned_user_rates_mbps
        < float(qos_threshold_mbps),
        axis=1
    )

    equal_qos_violation_counterfactual = np.mean(
        equal_user_rates_mbps
        < float(qos_threshold_mbps),
        axis=1
    )
    # ============================================================
    # QoS rescue and new-violation analysis
    # ============================================================

    qos_threshold_value = float(
        qos_threshold_mbps
    )

    # Equal allocation下所有违反QoS的“用户-时间”位置
    equal_violation_mask = (
        equal_user_rates_mbps
        < qos_threshold_value
    )

    # Learned allocation下所有违反QoS的“用户-时间”位置
    learned_violation_mask = (
        learned_user_rates_mbps
        < qos_threshold_value
    )

    # Equal下违约，但learned下被救回
    qos_rescued_mask = (
        equal_violation_mask
        & (~learned_violation_mask)
    )

    # Equal下没有违约，但learned下新产生违约
    qos_new_violation_mask = (
        (~equal_violation_mask)
        & learned_violation_mask
    )

    equal_violation_count_total = int(
        np.sum(
            equal_violation_mask
        )
    )

    learned_violation_count_total = int(
        np.sum(
            learned_violation_mask
        )
    )

    qos_rescued_count = int(
        np.sum(
            qos_rescued_mask
        )
    )

    qos_new_violation_count = int(
        np.sum(
            qos_new_violation_mask
        )
    )

    # Equal allocation下的违约中，有多少比例被PPO救回
    if equal_violation_count_total > 0:
        qos_rescue_rate = float(
            qos_rescued_count
            / equal_violation_count_total
        )
    else:
        qos_rescue_rate = np.nan

    # Learned allocation是否也制造了新的违约
    equal_non_violation_count_total = int(
        np.sum(
            ~equal_violation_mask
        )
    )

    if equal_non_violation_count_total > 0:
        qos_new_violation_rate = float(
            qos_new_violation_count
            / equal_non_violation_count_total
        )
    else:
        qos_new_violation_rate = np.nan

    # 净减少的QoS违约数量
    qos_net_reduction_count = int(
        equal_violation_count_total
        - learned_violation_count_total
    )


    # 每一步实际最差用户
    worst_user_indices = np.argmin(
        learned_user_rates_mbps,
        axis=1
    )

    step_indices = np.arange(
        num_evaluation_steps
    )

    # ============================================================
    # Equal-allocation weakest-user analysis
    # ============================================================

    step_indices = np.arange(
        num_evaluation_steps
    )

    # 先根据 equal-allocation 反事实结果，
    # 确定在没有 learned allocation 帮助时最弱的用户。
    equal_worst_user_indices = np.argmin(
        equal_user_rates_mbps,
        axis=1
    )

    # PPO 实际给这个“equal条件下最弱用户”的功率比例
    equal_worst_user_learned_fraction = (
        learned_allocation_fractions[
            step_indices,
            equal_worst_user_indices
        ]
    )

    # 均匀分配参考值：5个用户时为0.2
    equal_fraction_reference = float(
        1.0 / num_evaluation_users
    )

    # PPO 是否给原本最弱用户额外功率
    equal_worst_user_allocation_advantage = (
        equal_worst_user_learned_fraction
        - equal_fraction_reference
    )

    # 同一个用户，在learned allocation下的速率
    equal_worst_user_rate_under_learned = (
        learned_user_rates_mbps[
            step_indices,
            equal_worst_user_indices
        ]
    )

    # 同一个用户，在equal allocation下的速率
    equal_worst_user_rate_under_equal = (
        equal_user_rates_mbps[
            step_indices,
            equal_worst_user_indices
        ]
    )

    # PPO allocation 对原弱用户产生的速率增益
    equal_worst_user_rate_gain_mbps = (
        equal_worst_user_rate_under_learned
        - equal_worst_user_rate_under_equal
    )
    altitude_hold_ratio = float(
        np.mean(
            data["altitude_hold_active"].astype(
                np.float32
            )
        )
    )

    valid_altitude_hold_targets = (
        data["altitude_hold_target"][
            np.isfinite(
                data["altitude_hold_target"]
            )
        ]
    )

    if len(valid_altitude_hold_targets) > 0:
        mean_altitude_hold_target = float(
            np.mean(
                valid_altitude_hold_targets
            )
        )
    else:
        mean_altitude_hold_target = np.nan
    if len(data["reward"]) == 0:
        raise RuntimeError(
            "Evaluation produced no valid steps."
        )
    
    final_rates = data["rates"][-1]
    mean_rates = np.mean(data["rates"], axis=0)
    steps = np.arange(
        1,
        len(data["reward"]) + 1
    )

    total_distance = float(np.sum(np.linalg.norm(np.diff(uav_history, axis=0), axis=1)))

    # ============================================================
    # UAV steady-state motion diagnostics
    # ============================================================

    # 每一步 UAV 的三维位移
    step_displacements = np.linalg.norm(
        np.diff(
            uav_history,
            axis=0
        ),
        axis=1
    )

    # 小幅移动阈值：每步小于 0.5 m
    SMALL_MOTION_THRESHOLD = 0.5

    small_motion_ratio = float(
        np.mean(
            step_displacements
            < SMALL_MOTION_THRESHOLD
        )
    )

    # 最后 50 步的总移动距离
    late_stage_window = min(
        50,
        len(step_displacements)
    )

    late_stage_distance = float(
        np.sum(
            step_displacements[
                -late_stage_window:
            ]
        )
    )

    # 最后 50 步的平均每步移动距离
    late_stage_mean_displacement = float(
        np.mean(
            step_displacements[
                -late_stage_window:
            ]
        )
    )

    # 最后 50 步最大单步移动距离
    late_stage_max_displacement = float(
        np.max(
            step_displacements[
                -late_stage_window:
            ]
        )
    )

    print("\n" + "=" * 64)
    print("PPO evaluation summary")
    print("=" * 64)
    print(f"Episode steps: {len(steps)}")
    print(f"Mean reward: {np.mean(data['reward']):.4f}")
    print(f"Mean EE: {np.mean(data['ee']) / 1e6:.4f} Mbit/J")
    print(f"Mean total power: {np.mean(data['total_power']):.4f} W")
    print(f"Episode-mean minimum rate: {np.mean(data['min_rate']):.4f} Mbps")
    print(f"Episode-mean average rate: {np.mean(data['avg_rate']):.4f} Mbps")
    print(f"Episode-mean Jain index: {np.mean(data['jain']):.4f}")
    print(f"Episode-mean tracking distance: {np.mean(data['tracking_distance']):.4f} m")
    print(f"Evaluation QoS violation ratio: {np.mean(data['qos_violation_ratio']):.4f}")
    print(f"Total UAV travel distance: {total_distance:.4f} m")
    print(
        f"Training channel model: "
        f"{TRAINING_CHANNEL_MODEL}"
    )

    print(
        f"Evaluation channel model: "
        f"{EVALUATION_CHANNEL_MODEL}"
    )
    print(
        f"Small-motion ratio "
        f"(< {SMALL_MOTION_THRESHOLD:.1f} m/step): "
        f"{small_motion_ratio:.4f}"
    )

    print(
        f"Last {late_stage_window} steps travel distance: "
        f"{late_stage_distance:.4f} m"
    )

    print(
        f"Last {late_stage_window} steps mean displacement: "
        f"{late_stage_mean_displacement:.4f} m/step"
    )

    print(
        f"Last {late_stage_window} steps maximum displacement: "
        f"{late_stage_max_displacement:.4f} m/step"
    )
    print(
        f"Altitude-hold active ratio: "
        f"{altitude_hold_ratio:.4f}"
    )

    if np.isfinite(
        mean_altitude_hold_target
    ):
        print(
            f"Mean altitude-hold target: "
            f"{mean_altitude_hold_target:.4f} m"
        )
    else:
        print(
            "Mean altitude-hold target: "
            "not activated"
        )

    print(
        f"Mean absolute raw vertical action: "
        f"{np.mean(np.abs(data['vertical_action_raw'])):.4f}"
    )

    print(
        f"Mean absolute applied vertical action: "
        f"{np.mean(np.abs(data['vertical_action_applied'])):.4f}"
    )
    print("\n" + "-" * 64)
    print("Power-allocation counterfactual")
    print(
        f"Equal-allocation total QoS violations: "
        f"{equal_violation_count_total}"
    )

    print(
        f"Learned-allocation total QoS violations: "
        f"{learned_violation_count_total}"
    )

    print(
        f"QoS violations rescued by learned allocation: "
        f"{qos_rescued_count}"
    )

    if np.isfinite(qos_rescue_rate):
        print(
            f"QoS rescue rate: "
            f"{qos_rescue_rate:.4f}"
        )
    else:
        print(
            "QoS rescue rate: not applicable "
            "(equal allocation produced no violations)"
        )

    print(
        f"New QoS violations introduced by learned allocation: "
        f"{qos_new_violation_count}"
    )

    if np.isfinite(qos_new_violation_rate):
        print(
            f"New QoS violation rate: "
            f"{qos_new_violation_rate:.4f}"
        )

    print(
        f"Net QoS violation reduction count: "
        f"{qos_net_reduction_count}"
    )
    print("-" * 64)

    print(
        f"Learned-allocation mean minimum rate: "
        f"{np.mean(learned_min_rates_mbps):.4f} Mbps"
    )

    print(
        f"Equal-allocation mean minimum rate: "
        f"{np.mean(equal_min_rates_mbps):.4f} Mbps"
    )

    print(
        f"Mean minimum-rate improvement: "
        f"{np.mean(min_rate_improvement_mbps):.4f} Mbps"
    )

    print(
        f"Learned-allocation QoS violation ratio: "
        f"{np.mean(learned_qos_violation_counterfactual):.4f}"
    )

    print(
        f"Equal-allocation QoS violation ratio: "
        f"{np.mean(equal_qos_violation_counterfactual):.4f}"
    )

    print(
        f"Learned-allocation mean Jain index: "
        f"{np.mean(learned_jain_counterfactual):.4f}"
    )

    print(
        f"Equal-allocation mean Jain index: "
        f"{np.mean(equal_jain_counterfactual):.4f}"
    )

    print(
        f"Fraction of steps where learned allocation "
        f"improves minimum rate: "
        f"{np.mean(min_rate_improvement_mbps > 0.0):.4f}"
    )

    print(
        f"Mean allocation advantage for the "
        f"equal-allocation weakest user: "
        f"{np.mean(equal_worst_user_allocation_advantage):.4f}"
    )

    print(
        f"Mean rate gain for the "
        f"equal-allocation weakest user: "
        f"{np.mean(equal_worst_user_rate_gain_mbps):.4f} Mbps"
    )

    print(
        f"Fraction of steps where the equal-allocation "
        f"weakest user receives extra power: "
        f"{np.mean(equal_worst_user_allocation_advantage > 0.0):.4f}"
    )

    print(
        f"Fraction of steps where the equal-allocation "
        f"weakest user gains rate: "
        f"{np.mean(equal_worst_user_rate_gain_mbps > 0.0):.4f}"
    )
    print(f"Episode-mean per-user rates: {np.round(mean_rates, 2)} Mbps")
    print(f"Final per-user rates: {np.round(final_rates, 2)} Mbps")

    # ============================================================
    # Figure 1: Spatial trajectory (paper-style)
    # ============================================================
    fig = plt.figure(figsize=(20, 7))

    # Keep a fixed user-color mapping across all evaluation figures.
    user_colors = [
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    uav_color = "tab:blue"
    centroid_color = "tab:pink"
    building_color = "lightsteelblue"
    target_color = "tab:green"

    # ------------------------------------------------------------
    # 1A. 3D trajectory: UAV, users and full building volumes
    # ------------------------------------------------------------
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(
        uav_history[:, 0],
        uav_history[:, 1],
        uav_history[:, 2],
        linewidth=2.4,
        color=uav_color,
        label="UAV trajectory",
    )

    for user_idx in range(env.num_users):
        ax3d.plot(
            user_positions_history[:, user_idx, 0],
            user_positions_history[:, user_idx, 1],
            np.zeros(user_positions_history.shape[0]),
            linewidth=1.5,
            color=user_colors[user_idx],
            label=f"User {user_idx + 1}",
        )

    for building in env.buildings:
        poly = Poly3DCollection(
            _building_vertices(building),
            alpha=0.28,
            edgecolor="slategray",
            facecolor=building_color,
        )
        ax3d.add_collection3d(poly)
        ax3d.text(
            (building["x_min"] + building["x_max"]) / 2,
            (building["y_min"] + building["y_max"]) / 2,
            building["height"] + 3,
            building["name"],
            ha="center",
        )

    ax3d.scatter(
        *uav_history[0],
        marker="o",
        s=65,
        color=uav_color,
        label="UAV start",
    )
    ax3d.scatter(
        *uav_history[-1],
        marker="x",
        s=90,
        color=uav_color,
        label="UAV end",
    )
    ax3d.set_xlim(0, env.MAP_LIMIT)
    ax3d.set_ylim(0, env.MAP_LIMIT)
    ax3d.set_zlim(0, env.MAX_ALTITUDE)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Altitude (m)")
    ax3d.set_title("3D UAV and User Trajectories")
    ax3d.view_init(elev=26, azim=-58)
    ax3d.legend(fontsize=8, loc="upper left")

    ax_top = fig.add_subplot(1, 3, 2)

    # ============================================================
    # Time-coloured UAV trajectory
    # ============================================================

    uav_xy = np.asarray(
        uav_history[:, :2],
        dtype=np.float64
    )

    trajectory_steps = np.arange(
        len(uav_xy),
        dtype=np.float64
    )

    # 将相邻轨迹点组成线段：
    # [point_0, point_1], [point_1, point_2], ...
    trajectory_segments = np.stack(
        [
            uav_xy[:-1],
            uav_xy[1:]
        ],
        axis=1
    )

    time_normalization = Normalize(
        vmin=float(trajectory_steps[0]),
        vmax=float(trajectory_steps[-1])
    )

    trajectory_collection = LineCollection(
        trajectory_segments,
        cmap="viridis",
        norm=time_normalization,
        linewidth=2.8,
        zorder=4
    )

    # 每条线段使用其起始 timestep 着色
    trajectory_collection.set_array(
        trajectory_steps[:-1]
    )

    ax_top.add_collection(
        trajectory_collection
    )

    # 为图例增加一个代理对象。
    # 实际渐变轨迹由 LineCollection 绘制。
    ax_top.plot(
        [],
        [],
        linewidth=2.8,
        color="tab:blue",
        label="UAV trajectory (time-coloured)"
    )

    # ============================================================
    # Direction arrows distributed by travelled distance
    # ============================================================

    uav_xy = np.asarray(
        uav_history[:, :2],
        dtype=np.float64
    )

    segment_lengths = np.linalg.norm(
        np.diff(
            uav_xy,
            axis=0
        ),
        axis=1
    )

    cumulative_distance = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(segment_lengths)
        ]
    )

    total_xy_distance = float(
        cumulative_distance[-1]
    )

    # 整条轨迹只画 4 个方向箭头
    num_direction_arrows = 4

    if total_xy_distance > 1e-6:

        arrow_distances = np.linspace(
            0.12 * total_xy_distance,
            0.88 * total_xy_distance,
            num_direction_arrows
        )

        for target_distance in arrow_distances:

            idx = int(
                np.searchsorted(
                    cumulative_distance,
                    target_distance
                )
            )

            idx = int(
                np.clip(
                    idx,
                    1,
                    len(uav_xy) - 1
                )
            )

            # 使用前后若干步确定局部移动方向，
            # 避免单步位移太短导致箭头堆叠。
            start_idx = max(
                0,
                idx - 3
            )

            end_idx = min(
                len(uav_xy) - 1,
                idx + 3
            )

            start_point = uav_xy[
                start_idx
            ]

            end_point = uav_xy[
                end_idx
            ]

            direction_vector = (
                end_point
                - start_point
            )

            direction_norm = float(
                np.linalg.norm(
                    direction_vector
                )
            )

            # 局部几乎没有移动时不画箭头
            if direction_norm < 1.0:
                continue

            unit_direction = (
                direction_vector
                / direction_norm
            )

            # 固定箭头长度，避免慢速区域出现极短箭头
            arrow_length = 18.0

            arrow_start = (
                uav_xy[idx]
                - 0.5
                * arrow_length
                * unit_direction
            )

            arrow_end = (
                uav_xy[idx]
                + 0.5
                * arrow_length
                * unit_direction
            )

            ax_top.annotate(
                "",
                xy=(
                    arrow_end[0],
                    arrow_end[1]
                ),
                xytext=(
                    arrow_start[0],
                    arrow_start[1]
                ),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=uav_color,
                    lw=1.4,
                    mutation_scale=13
                ),
                zorder=6
            )

    # ============================================================
    # ============================================================
    # Time markers every 25 evaluation steps
    # ============================================================

    TIME_MARKER_INTERVAL = 25

    time_marker_indices = np.arange(
        TIME_MARKER_INTERVAL,
        len(uav_xy),
        TIME_MARKER_INTERVAL,
        dtype=int
    )

    # 如果最终step不是25的整数倍，额外保留最后一步。
    if (
        len(uav_xy) > 1
        and (
            len(uav_xy) - 1
            not in time_marker_indices
        )
    ):
        time_marker_indices = np.append(
            time_marker_indices,
            len(uav_xy) - 1
        )

    marker_colours = plt.cm.viridis(
        time_normalization(
            time_marker_indices
        )
    )

    ax_top.scatter(
        uav_xy[
            time_marker_indices,
            0
        ],
        uav_xy[
            time_marker_indices,
            1
        ],
        c=marker_colours,
        s=42,
        marker="o",
        edgecolors="black",
        linewidths=0.7,
        zorder=8,
        label="25-step time markers"
    )

    for marker_idx in time_marker_indices:

        # uav_history[0] 是初始状态，
        # 因此索引值也对应评估 timestep。
        marker_label = str(
            int(marker_idx)
        )

        ax_top.annotate(
            marker_label,
            xy=(
                uav_xy[marker_idx, 0],
                uav_xy[marker_idx, 1]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.72
            ),
            zorder=9
        )

    ax_top.plot(
        centroid_history[:, 0],
        centroid_history[:, 1],
        linestyle="--",
        linewidth=2.0,
        color=centroid_color,
        label="Moving centroid",
    )
    ax_top.scatter(
        *centroid_history[0],
        marker="o",
        s=75,
        color=centroid_color,
        label="Centroid start",
        zorder=5,
    )
    ax_top.scatter(
        *centroid_history[-1],
        marker="*",
        s=150,
        color=centroid_color,
        label="Centroid end",
        zorder=5,
    )
    ax_top.annotate(
        "Centroid start",
        centroid_history[0],
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
    )
    ax_top.annotate(
        "Centroid end",
        centroid_history[-1],
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
    )

    for user_idx in range(env.num_users):
        ax_top.plot(
            user_positions_history[:, user_idx, 0],
            user_positions_history[:, user_idx, 1],
            linewidth=1.5,
            color=user_colors[user_idx],
            label=f"User {user_idx + 1}",
        )
        ax_top.scatter(
            *user_positions_history[0, user_idx],
            marker="o",
            s=38,
            color=user_colors[user_idx],
        )
        ax_top.scatter(
            *user_positions_history[-1, user_idx],
            marker="s",
            s=38,
            color=user_colors[user_idx],
        )

    for building in env.buildings:
        ax_top.add_patch(
            Rectangle(
                (building["x_min"], building["y_min"]),
                building["x_max"] - building["x_min"],
                building["y_max"] - building["y_min"],
                alpha=0.35,
                facecolor=building_color,
                edgecolor="slategray",
            )
        )
        ax_top.text(
            (building["x_min"] + building["x_max"]) / 2,
            (building["y_min"] + building["y_max"]) / 2,
            f"{building['name']}\nH={building['height']:.0f} m",
            ha="center",
            va="center",
        )

    ax_top.scatter(
        *uav_history[0, :2],
        marker="o",
        s=70,
        color=plt.cm.viridis(0.0),
        edgecolors="black",
        linewidths=0.8,
        label="UAV start",
        zorder=10,
    )
    ax_top.scatter(
        *uav_history[-1, :2],
        marker="x",
        s=90,
        color=plt.cm.viridis(1.0),
        linewidths=2.2,
        label="UAV end",
        zorder=10,
    )
    if evacuation_target is not None:
        target_label = (
            "Evacuation target"
            if scenario == "evacuation"
            else "Group destination"
        )

        ax_top.scatter(
            *evacuation_target,
            marker="*",
            s=150,
            color=target_color,
            label=target_label,
            zorder=5,
        )
    ax_top.set_xlim(0, env.MAP_LIMIT)
    ax_top.set_ylim(0, env.MAP_LIMIT)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title(
        "Top View (X–Y) with Temporal Progression"
    )

    ax_top.grid(True)

    ax_top.legend(
        fontsize=8,
        loc="lower right",
        ncol=2
    )

    time_colorbar = fig.colorbar(
        trajectory_collection,
        ax=ax_top,
        fraction=0.046,
        pad=0.04
    )

    time_colorbar.set_label(
        "Evaluation Step"
    )

    

    # ------------------------------------------------------------
    # 1C. Side view in the real X-Z coordinate plane
    # ------------------------------------------------------------
    ax_side = fig.add_subplot(1, 3, 3)

    # UAV trajectory projected onto the actual X-Z plane
    ax_side.plot(
        uav_history[:, 0],
        uav_history[:, 2],
        linewidth=2.4,
        color=uav_color,
        label="UAV altitude profile",
    )

    ax_side.scatter(
        uav_history[0, 0],
        uav_history[0, 2],
        marker="o",
        s=55,
        color=uav_color,
        label="UAV start",
    )

    ax_side.scatter(
        uav_history[-1, 0],
        uav_history[-1, 2],
        marker="x",
        s=75,
        color=uav_color,
        label="UAV end",
    )

    # Buildings projected onto the actual X-Z plane
    for building in env.buildings:
        x_min = float(building["x_min"])
        x_max = float(building["x_max"])
        building_height = float(building["height"])

        ax_side.add_patch(
            Rectangle(
                (x_min, 0.0),
                x_max - x_min,
                building_height,
                alpha=0.35,
                facecolor=building_color,
                edgecolor="slategray",
            )
        )

        ax_side.text(
            (x_min + x_max) / 2.0,
            building_height + 2.0,
            building["name"],
            ha="center",
            va="bottom",
        )

    ax_side.axhline(
        env.MIN_ALTITUDE,
        linestyle=":",
        linewidth=1.3,
        label="Minimum altitude",
    )

    ax_side.axhline(
        env.MAX_ALTITUDE,
        linestyle=":",
        linewidth=1.3,
        label="Maximum altitude",
    )

    # The key lines: real map X-axis, fixed to 0–500 m
    ax_side.set_xlim(
        0.0,
        float(env.MAP_LIMIT),
    )

    ax_side.set_ylim(
        0.0,
        float(env.MAX_ALTITUDE) + 5.0,
    )

    ax_side.set_xlabel(
        "X Position (m)"
    )

    ax_side.set_ylabel(
        "Altitude (m)"
    )

    ax_side.set_title(
        "Side View (X-Z)"
    )

    ax_side.grid(True)

    ax_side.legend(
        fontsize=8,
        loc="lower right",
    )

    fig.tight_layout()
    spatial_path = f"{output_prefix}_spatial_trajectory.png"
    fig.savefig(spatial_path, dpi=150, bbox_inches="tight",pad_inches=0.05,)
    plt.close(fig)
    # ============================================================
    # V5.3: Per-user velocity
    # ============================================================
    user_velocities = np.asarray(
        data["user_velocities"],
        dtype=np.float32
    )

    speed_magnitudes = np.linalg.norm(
        user_velocities,
        axis=2
    )

    fig_velocity, ax_velocity = plt.subplots(
        figsize=(10, 5.5)
    )

    for user_idx in range(
        speed_magnitudes.shape[1]
    ):
        ax_velocity.plot(
            steps,
            speed_magnitudes[:, user_idx],
            linewidth=1.8,
            label=f"User {user_idx + 1}"
        )

    ax_velocity.set_title(
        "Per-User Speed Magnitude Over Time"
    )
    ax_velocity.set_xlabel("Step")
    ax_velocity.set_ylabel("Speed (m/s)")
    ax_velocity.grid(True)
    ax_velocity.legend(ncol=2)

    fig_velocity.tight_layout()

    velocity_path = (
        f"{output_prefix}_user_velocity.png"
    )

    fig_velocity.savefig(
        velocity_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig_velocity)
    # ============================================================
    # Figure 2: Communication and tracking
    # ============================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 6.5)
    )

    # ------------------------------------------------------------
    # Left: per-user and worst-user data rates
    # ------------------------------------------------------------
    ax = axes[0]

    rate_smoothing_window = 5

    for user_idx in range(
        data["rates"].shape[1]
    ):
        raw_rate = data["rates"][:, user_idx]

        smoothed_rate = moving_average(
            raw_rate,
            window=rate_smoothing_window
        )

        # Raw instantaneous rate
        ax.plot(
            steps,
            raw_rate,
            linewidth=0.7,
            alpha=0.20,
            color=user_colors[user_idx]
        )

        # Smoothed trend
        ax.plot(
            steps,
            smoothed_rate,
            linewidth=2.0,
            color=user_colors[user_idx],
            label=f"User {user_idx + 1}"
        )

    worst_user_rate_raw = np.min(
        data["rates"],
        axis=1
    )

    worst_user_rate_smoothed = moving_average(
        worst_user_rate_raw,
        window=rate_smoothing_window
    )

    # Faint raw worst-user rate
    ax.plot(
        steps,
        worst_user_rate_raw,
        linewidth=0.8,
        alpha=0.20,
        color="black"
    )

    # Smoothed worst-user rate
    ax.plot(
        steps,
        worst_user_rate_smoothed,
        linewidth=2.8,
        color="black",
        label="Worst-user rate",
        zorder=6
    )

    ax.axhline(
        qos_threshold_mbps,
        linestyle="--",
        linewidth=1.5,
        color="dimgray",
        label=(
            f"QoS threshold "
            f"({qos_threshold_mbps:.0f} Mbps)"
        )
    )

    ax.set_title(
        "Per-User Data Rates "
        "(Raw and 5-Step Moving Average)"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Rate (Mbps)")
    ax.grid(True)
    ax.legend(ncol=2)

    # ------------------------------------------------------------
    # Right: tracking distances
    # ------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        steps,
        data["tracking_distance"],
        linewidth=2.2,
        label="Distance to moving centroid"
    )

    ax.plot(
        steps,
        data["max_user_distance"],
        linewidth=1.9,
        label="Maximum UAV-to-user distance"
    )

    ax.plot(
        steps,
        data["mean_user_distance"],
        linewidth=1.6,
        linestyle="--",
        label="Mean UAV-to-user distance"
    )

    ax.set_title(
        "Tracking and Edge-User Distance"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    dynamic_path = (
        f"{output_prefix}_dynamic_performance.png"
    )

    fig.savefig(
        dynamic_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)
    # ============================================================
    # LoS/NLoS heatmap
    # ============================================================
    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    heatmap = ax.imshow(
        data["los_flags"].T.astype(int),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[
            steps[0],
            steps[-1],
            -0.5,
            data["los_flags"].shape[1] - 0.5
        ]
    )

    ax.set_yticks(
        np.arange(
            data["los_flags"].shape[1]
        )
    )

    ax.set_yticklabels(
        [
            f"User {idx + 1}"
            for idx in range(
                data["los_flags"].shape[1]
            )
        ]
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("User")
    ax.set_title(
        "LoS/NLoS Status Heatmap"
    )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax
    )

    colorbar.set_ticks(
        [0, 1]
    )

    colorbar.set_ticklabels(
        ["NLoS", "LoS"]
    )

    fig.tight_layout()

    heatmap_path = (
        f"{output_prefix}_los_nlos_heatmap.png"
    )

    fig.savefig(
        heatmap_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)
    # ============================================================
    # Figure 3: Energy and compact reward diagnostics
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    ax = axes[0]
    ax.plot(
        steps,
        data["ee"] / 1e6,
        linewidth=2.2,
        label="PPO EE",
    )
    ax.plot(
        np.arange(1, len(baseline["ee"]) + 1),
        baseline["ee"] / 1e6,
        linestyle="--",
        linewidth=1.9,
        label="Initial-centroid hover EE",
    )
    ax.set_title("Energy Efficiency")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy Efficiency (Mbit/J)")
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    ax.plot(
        steps,
        data["total_power"],
        linewidth=2.2,
        label="Total power",
    )
    ax.plot(
        steps,
        data["aero_power"],
        linewidth=1.7,
        label="Aerodynamic power",
    )
    ax.plot(
        steps,
        data["transmit_power"],
        linewidth=1.7,
        label="Transmit power",
    )
    ax.plot(
        steps,
        data["electronic_power"],
        linewidth=1.5,
        linestyle="--",
        label="Electronic power",
    )
    ax.set_title("Power Decomposition")
    ax.set_xlabel("Step")
    ax.set_ylabel("Power (W)")
    ax.grid(True)
    ax.legend()

    ax = axes[2]
    tracking_reward = (
        data["reward_tracking_distance"]
        + data["reward_tracking_progress"]
    )
    compact_reward_series = [
        ("EE", data["reward_ee"]),
        ("Worst-user QoS", data["reward_worst_qos"]),
        ("Tracking", tracking_reward),
        ("Movement", data["reward_movement"]),
    ]
    for label, values in compact_reward_series:
        ax.plot(steps, values, linewidth=1.7, label=label)
    ax.set_title("Selected Reward Components")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Contribution")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    resource_path = f"{output_prefix}_energy_resource_control.png"
    fig.savefig(resource_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # ============================================================
    # Figure 4: Learned power-allocation behaviour
    # ============================================================

    fig_allocation, axes_allocation = plt.subplots(
        1,
        3,
        figsize=(21, 6.5)
    )

    # ------------------------------------------------------------
    # 4A. Per-user learned allocation fractions
    # ------------------------------------------------------------

    ax = axes_allocation[0]

    allocation_heatmap = ax.imshow(
        learned_allocation_fractions.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[
            steps[0],
            steps[-1],
            -0.5,
            num_evaluation_users - 0.5
        ]
    )

    ax.set_yticks(
        np.arange(
            num_evaluation_users
        )
    )

    ax.set_yticklabels(
        [
            f"User {user_idx + 1}"
            for user_idx in range(
                num_evaluation_users
            )
        ]
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("User")
    ax.set_title(
        "Learned Per-User "
        "Power-Allocation Fractions"
    )

    allocation_colorbar = fig_allocation.colorbar(
        allocation_heatmap,
        ax=ax
    )

    allocation_colorbar.set_label(
        "Fraction of Total Transmit Power"
    )

    # ------------------------------------------------------------
    # 4B. Same-state minimum-rate counterfactual
    # ------------------------------------------------------------

    ax = axes_allocation[1]

    ax.plot(
        steps,
        moving_average(
            learned_min_rates_mbps,
            window=5
        ),
        linewidth=2.3,
        label="Learned allocation"
    )

    ax.plot(
        steps,
        moving_average(
            equal_min_rates_mbps,
            window=5
        ),
        linewidth=2.1,
        linestyle="--",
        label="Equal allocation"
    )

    ax.axhline(
        float(qos_threshold_mbps),
        linestyle=":",
        linewidth=1.6,
        label=(
            f"QoS threshold "
            f"({qos_threshold_mbps:.0f} Mbps)"
        )
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Worst-User Rate (Mbps)")
    ax.set_title(
        "Same-State Counterfactual:\n"
        "Learned vs Equal Allocation"
    )
    ax.grid(True)
    ax.legend()

    # ------------------------------------------------------------
    # 4C. Allocation response for the current worst user
    # ------------------------------------------------------------

    ax = axes_allocation[2]

    ax.plot(
        steps,
        moving_average(
            equal_worst_user_learned_fraction,
            window=5
        ),
        linewidth=2.2,
        label="Worst-user learned fraction"
    )

    ax.axhline(
        1.0 / num_evaluation_users,
        linestyle="--",
        linewidth=1.7,
        label=(
            f"Equal-allocation reference "
            f"({1.0 / num_evaluation_users:.2f})"
        )
    )

    ax.fill_between(
        steps,
        1.0 / num_evaluation_users,
        equal_worst_user_learned_fraction,
        where=(
            equal_worst_user_learned_fraction
            > 1.0 / num_evaluation_users
        ),
        alpha=0.20,
        label="Extra power to worst user"
    )

    ax.set_xlabel("Step")
    ax.set_ylabel(
        "Fraction of Total Transmit Power"
    )
    ax.set_title(
        "Power Assigned to the "
        "the Equal-Allocation Weakest User"
    )
    ax.grid(True)
    ax.legend()

    fig_allocation.tight_layout()

    allocation_path = (
        f"{output_prefix}_"
        f"power_allocation_analysis.png"
    )

    fig_allocation.savefig(
        allocation_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig_allocation)
    result_data = {
        "experiment_name": EXPERIMENT_NAME,
        "model_group": MODEL_GROUP,
        "power_allocation_mode":
        POWER_ALLOCATION_MODE,
        "training_mobility": TRAINING_MOBILITY,
        "evaluation_scenario": scenario,
        "model_path": MODEL_PATH,
        "channel_model": channel_model,
        "use_interference": use_interference,
        "interference_coupling": interference_coupling,
        "seed": seed,
        "configured_max_steps": max_steps,
        "actual_episode_steps": len(steps),
        "total_uav_travel_distance": total_distance,
        "small_motion_threshold": SMALL_MOTION_THRESHOLD,
        "small_motion_ratio": small_motion_ratio,
        "late_stage_window": late_stage_window,
        "late_stage_distance": late_stage_distance,
        "late_stage_mean_displacement":
            late_stage_mean_displacement,
        "late_stage_max_displacement":
            late_stage_max_displacement,
        "altitude_hold_ratio":
            altitude_hold_ratio,

        "mean_altitude_hold_target":
            mean_altitude_hold_target,
        "step_displacements":
            step_displacements.copy(),

        "uav_history": uav_history,
        "user_positions_history": user_positions_history,

        "centroid_history": centroid_history,
        "initial_users": initial_users,
        "initial_centroid": initial_centroid,
        "evacuation_target": evacuation_target,
        "learned_allocation_fractions":
            learned_allocation_fractions,

        "equal_allocation_fractions":
            equal_allocation_fractions,

        "equal_power_allocation":
            equal_power_allocation,

        "equal_user_sinrs":
            equal_user_sinrs,

        "equal_interference_powers":
            equal_interference_powers,

        "equal_user_rates_mbps":
            equal_user_rates_mbps,

        "learned_min_rates_mbps":
            learned_min_rates_mbps,

        "equal_min_rates_mbps":
            equal_min_rates_mbps,

        "min_rate_improvement_mbps":
            min_rate_improvement_mbps,
        "equal_violation_mask":
            equal_violation_mask,

        "learned_violation_mask":
            learned_violation_mask,

        "qos_rescued_mask":
            qos_rescued_mask,

        "qos_new_violation_mask":
            qos_new_violation_mask,

        "equal_violation_count_total":
            equal_violation_count_total,

        "learned_violation_count_total":
            learned_violation_count_total,

        "qos_rescued_count":
            qos_rescued_count,

        "qos_new_violation_count":
            qos_new_violation_count,

        "qos_rescue_rate":
            qos_rescue_rate,

        "qos_new_violation_rate":
            qos_new_violation_rate,

        "qos_net_reduction_count":
            qos_net_reduction_count,
        

        "learned_jain_counterfactual":
            learned_jain_counterfactual,

        "equal_jain_counterfactual":
            equal_jain_counterfactual,

        "learned_qos_violation_counterfactual":
            learned_qos_violation_counterfactual,

        "equal_qos_violation_counterfactual":
            equal_qos_violation_counterfactual,

        "equal_worst_user_indices":
            equal_worst_user_indices,

        "equal_worst_user_learned_fraction":
            equal_worst_user_learned_fraction,

        "equal_worst_user_allocation_advantage":
            equal_worst_user_allocation_advantage,

        "equal_worst_user_rate_under_learned":
            equal_worst_user_rate_under_learned,

        "equal_worst_user_rate_under_equal":
            equal_worst_user_rate_under_equal,

        "equal_worst_user_rate_gain_mbps":
            equal_worst_user_rate_gain_mbps,
        "qos_threshold_mbps": qos_threshold_mbps,
        "training_channel_model":
            TRAINING_CHANNEL_MODEL,

        "evaluation_channel_model":
            EVALUATION_CHANNEL_MODEL,
        "baseline": baseline,
        **data,
    }
    result_path = f"{output_prefix}_data.npy"
    np.save(result_path, result_data, allow_pickle=True)

    print("\nGenerated files:")
    print(f"- {spatial_path}")
    print(f"- {dynamic_path}")
    print(f"- {heatmap_path}")
    print(f"- {resource_path}")
    print(f"- {result_path}")
    print(f"- {velocity_path}")
    print(f"- {allocation_path}")
    env.close()
    return result_data

def evaluate_policy_at_rho(
    model_path: str,
    scenario: str,
    rho: float,
    seed: int = 42,
    max_steps: int = 200,
    power_allocation_mode: str = "learned",
) -> Dict[str, float]:
    """
    Lightweight evaluation for rho sensitivity analysis.

    The trained model is kept fixed while only the evaluation
    interference coefficient rho is changed.
    """

    if not 0.0 <= rho <= 1.0:
        raise ValueError(
            "rho must be between 0 and 1."
        )

    if (
        not os.path.exists(model_path)
        and not os.path.exists(model_path + ".zip")
    ):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    env = UAV6GMultiUserEnv(
        num_users=5,
        mode="general",
        scenario=scenario,
        use_interference=True,
        interference_coupling=rho,
        power_allocation_mode=power_allocation_mode,
    )

    model = PPO.load(
        model_path,
        device="cpu",
    )

    obs, _ = env.reset(seed=seed)

    ee_history = []
    min_rate_history = []
    avg_rate_history = []
    jain_history = []
    qos_history = []

    for _ in range(max_steps):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

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

    results = {
        "rho": float(rho),
        "mean_ee_mbit_j": float(
            np.mean(ee_history)
        ),
        "mean_min_rate_mbps": float(
            np.mean(min_rate_history)
        ),
        "mean_avg_rate_mbps": float(
            np.mean(avg_rate_history)
        ),
        "mean_jain": float(
            np.mean(jain_history)
        ),
        "mean_qos_violation": float(
            np.mean(qos_history)
        ),
        "episode_steps": len(ee_history),
    }

    return results

def run_rho_sensitivity(
    scenario: str,
    seed: int = 42,
    max_steps: int = 200,
):
    """
    Compare SINR-trained Group A and SNR-trained Group D
    under increasing residual interference.
    """

    scenario = scenario.lower().strip()

    output_dir = os.path.join(
        "./evaluations",
        "rho_sensitivity",
        scenario,
        f"seed{seed}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    group_a_results = []
    group_d_results = []

    print("\n" + "=" * 70)
    print(
        f"Rho sensitivity analysis: {scenario}"
    )
    print("=" * 70)

    for rho in RHO_VALUES:

        print(
            f"\nEvaluating rho = {rho:.2f}"
        )

        # --------------------------------------------------------
        # Group A: SINR-trained PPO
        # --------------------------------------------------------
        result_a = evaluate_policy_at_rho(
            model_path=GROUP_A_MODEL_PATH,
            scenario=scenario,
            rho=rho,
            seed=seed,
            max_steps=max_steps,
            power_allocation_mode="learned",
        )

        group_a_results.append(
            result_a
        )

        # --------------------------------------------------------
        # Group D: SNR-trained PPO
        # --------------------------------------------------------
        result_d = evaluate_policy_at_rho(
            model_path=GROUP_D_MODEL_PATH,
            scenario=scenario,
            rho=rho,
            seed=seed,
            max_steps=max_steps,
            power_allocation_mode="learned",
        )

        group_d_results.append(
            result_d
        )

        print(
            "Group A | "
            f"EE={result_a['mean_ee_mbit_j']:.2f} "
            "Mbit/J | "
            f"MinRate="
            f"{result_a['mean_min_rate_mbps']:.2f} "
            "Mbps | "
            f"Jain={result_a['mean_jain']:.4f} | "
            f"QoS="
            f"{result_a['mean_qos_violation']:.4f}"
        )

        print(
            "Group D | "
            f"EE={result_d['mean_ee_mbit_j']:.2f} "
            "Mbit/J | "
            f"MinRate="
            f"{result_d['mean_min_rate_mbps']:.2f} "
            "Mbps | "
            f"Jain={result_d['mean_jain']:.4f} | "
            f"QoS="
            f"{result_d['mean_qos_violation']:.4f}"
        )
        rho_values = np.asarray(
            RHO_VALUES,
            dtype=np.float64,
        )

        # Group A
        a_ee = np.asarray([
            r["mean_ee_mbit_j"]
            for r in group_a_results
        ])

        a_min_rate = np.asarray([
            r["mean_min_rate_mbps"]
            for r in group_a_results
        ])

        a_jain = np.asarray([
            r["mean_jain"]
            for r in group_a_results
        ])

        a_qos = np.asarray([
            r["mean_qos_violation"]
            for r in group_a_results
        ])

        # Group D
        d_ee = np.asarray([
            r["mean_ee_mbit_j"]
            for r in group_d_results
        ])

        d_min_rate = np.asarray([
            r["mean_min_rate_mbps"]
            for r in group_d_results
        ])

        d_jain = np.asarray([
            r["mean_jain"]
            for r in group_d_results
        ])

        d_qos = np.asarray([
            r["mean_qos_violation"]
            for r in group_d_results
        ])
    # ============================================================
    # Paper figure: 1 x 4 rho sensitivity analysis
    # ============================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, 4.3),
    )

    # ------------------------------------------------------------
    # (a) Energy efficiency
    # ------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        rho_values,
        a_ee,
        marker="o",
        linewidth=2.0,
        label="SINR-trained",
    )

    ax.plot(
        rho_values,
        d_ee,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="SNR-trained",
    )

    ax.set_xlabel(
        r"Interference coefficient $\rho$"
    )

    ax.set_ylabel(
        "Mean EE (Mbit/J)"
    )

    ax.set_title(
        "(a) Energy Efficiency"
    )

    ax.grid(True)

    # ------------------------------------------------------------
    # (b) Minimum user rate
    # ------------------------------------------------------------
    ax = axes[1]

    ax.plot(
        rho_values,
        a_min_rate,
        marker="o",
        linewidth=2.0,
        label="SINR-trained",
    )

    ax.plot(
        rho_values,
        d_min_rate,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="SNR-trained",
    )

    ax.set_xlabel(
        r"Interference coefficient $\rho$"
    )

    ax.set_ylabel(
        "Minimum Rate (Mbps)"
    )

    ax.set_title(
        "(b) Minimum User Rate"
    )

    ax.grid(True)

    # ------------------------------------------------------------
    # (c) Jain fairness
    # ------------------------------------------------------------
    ax = axes[2]

    ax.plot(
        rho_values,
        a_jain,
        marker="o",
        linewidth=2.0,
        label="SINR-trained",
    )

    ax.plot(
        rho_values,
        d_jain,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="SNR-trained",
    )

    ax.set_xlabel(
        r"Interference coefficient $\rho$"
    )

    ax.set_ylabel(
        "Jain Index"
    )

    ax.set_title(
        "(c) Fairness"
    )

    ax.grid(True)

    # ------------------------------------------------------------
    # (d) QoS violation
    # ------------------------------------------------------------
    ax = axes[3]

    ax.plot(
        rho_values,
        a_qos,
        marker="o",
        linewidth=2.0,
        label="SINR-trained",
    )

    ax.plot(
        rho_values,
        d_qos,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="SNR-trained",
    )

    ax.set_xlabel(
        r"Interference coefficient $\rho$"
    )

    ax.set_ylabel(
        "QoS Violation Ratio"
    )

    ax.set_title(
        "(d) QoS Violation"
    )

    ax.grid(True)

    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.tight_layout()

    figure_path = os.path.join(
        output_dir,
        f"rho_sensitivity_{scenario}.png",
    )

    fig.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    save_data = {
        "scenario": scenario,
        "seed": seed,
        "rho_values": rho_values,

        "group_a_training": "SINR rho=0.1",
        "group_d_training": "SNR rho=0",

        "group_a_results":
            group_a_results,

        "group_d_results":
            group_d_results,
    }

    data_path = os.path.join(
        output_dir,
        f"rho_sensitivity_{scenario}_data.npy",
    )

    np.save(
        data_path,
        save_data,
        allow_pickle=True,
    )

    print("\nGenerated rho-sensitivity files:")
    print(f"- {figure_path}")
    print(f"- {data_path}")

    return save_data


if __name__ == "__main__":
    # evaluate_and_visualize(
    #     num_users=5,
    #     mode="general",
    #     scenario="straggler",#"evacuation",#
    #     use_interference=True,
    #     interference_coupling=0.1,
    #     seed=TEST_SEED,
    #     max_steps=200,
    # )

    # run_rho_sensitivity(
    #     scenario="evacuation",
    #     seed=TEST_SEED,
    #     max_steps=200,
    # )

    # run_rho_sensitivity(
    #     scenario="straggler",
    #     seed=TEST_SEED,
    #     max_steps=200,
    # )
    evaluate_and_visualize(
        num_users=5,
        mode="general",
        scenario="random_walk",
        use_interference=True,
        interference_coupling=0.1,
        seed=42,
        max_steps=200,
        building_layout="default",
    )

    evaluate_and_visualize(
        num_users=5,
        mode="general",
        scenario="random_walk",
        use_interference=True,
        interference_coupling=0.1,
        seed=42,
        max_steps=200,
        building_layout="dense",
    )

