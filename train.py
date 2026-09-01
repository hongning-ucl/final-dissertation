import gymnasium as gym
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.env_1uav_5user_power import UAV6GMultiUserEnv

# 确保环境已注册，或者在这里直接 import
# from env_1uav_5user_power import UAV6GMultiUserEnv 
from stable_baselines3.common.callbacks import BaseCallback

from monitor_training import plot_training_metrics


class TrainingMonitorCallback(BaseCallback):
    """
    记录每个 episode 的核心通信与能耗指标。

    保存指标：
    1. episode reward
    2. mean energy efficiency
    3. minimum user rate
    4. average user rate
    5. mean total power
    6. Jain fairness index
    7. UAV travel distance
    8. QoS violation ratio
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        # 最终按 episode 保存的数据
        self.episode_rewards = []
        self.episode_mean_ee = []
        self.episode_min_rates = []
        self.episode_avg_rates = []
        self.episode_mean_power = []
        self.episode_jain_indices = []
        self.episode_travel_distances = []
        self.episode_qos_violation_ratios = []

        self.episode_mean_los_users = []
        self.episode_mean_nlos_users = []

        self.episode_mean_tracking_distances = []

        # 当前 episode 临时缓存
        self.current_rewards = []
        self.current_ee = []
        self.current_min_rates = []
        self.current_avg_rates = []
        self.current_power = []
        self.current_jain = []
        self.current_displacements = []
        self.current_qos_violation_ratios = []

        self.current_los_users = []
        self.current_nlos_users = []

        self.current_episode_tracking_distances = []

        self.episode_mobilities = []
        self.current_episode_mobility = None

        # ============================================================
        # V5.2 episode-level stability metrics
        # ============================================================
        self.episode_mean_power_allocation_changes = []
        self.episode_mean_rate_variations = []
        self.episode_p95_rate_variations = []

        # ============================================================
        # V5.2 temporary stability data
        # ============================================================
        self.current_power_allocation_changes = []
        self.current_rate_variations = []

        # 用于计算相邻 timestep 的 rate 变化
        self.previous_user_rates = None

    def _on_step(self) -> bool:
        """
        每一步从 info 中读取环境指标。
        DummyVecEnv 下 infos 是列表，这里使用第一个环境。
        """
        infos = self.locals.get("infos", [])

        if not infos:
            return True

        info = infos[0]

        active_mobility = info.get(
            "active_mobility"
        )

        if active_mobility is not None:
            if self.current_episode_mobility is None:
                self.current_episode_mobility = (
                    str(active_mobility)
                )

        # ============================================================
        # V5.2 power-allocation stability
        # ============================================================
        if "power_allocation_change_penalty" in info:
            self.current_power_allocation_changes.append(
                float(
                    info[
                        "power_allocation_change_penalty"
                    ]
                )
            )

        # ============================================================
        # V5.2 per-user rate variation
        # ============================================================
        if "user_rates_mbps" in info:

            current_user_rates = np.asarray(
                info["user_rates_mbps"],
                dtype=np.float32
            )

            if self.previous_user_rates is not None:

                rate_change = np.abs(
                    current_user_rates
                    - self.previous_user_rates
                )

                # 每一步五个用户平均绝对速率变化
                self.current_rate_variations.append(
                    float(
                        np.mean(rate_change)
                    )
                )

            self.previous_user_rates = (
                current_user_rates.copy()
            )

        # 仅在第一个 timestep 检查一次 tracking 数据链
        if self.num_timesteps == 1:
            print(
                "runtime use_interference:",
                info.get("use_interference")
            )

            print(
                "runtime interference_coupling:",
                info.get("interference_coupling")
            )

            print(
                "runtime user_snrs:",
                info.get("user_snrs")
            )

            print(
                "runtime user_sinrs:",
                info.get("user_sinrs")
            )

            print(
                "runtime interference powers:",
                info.get("user_interference_powers")
            )

            print(
                "runtime user rates Mbps:",
                info.get("user_rates_mbps")
            )
            print(
                "tracking_distance in info:",
                "tracking_distance" in info
            )
            print(
                "tracking_distance value:",
                info.get("tracking_distance")
            )
            print(
                "dist value:",
                info.get("dist")
            )

        # 当前 step reward
        rewards = self.locals.get("rewards")

        if "los_user_count" in info:
            self.current_los_users.append(
                float(info["los_user_count"])
            )

        if "nlos_user_count" in info:
            self.current_nlos_users.append(
                float(info["nlos_user_count"])
            )

        if rewards is not None:
            step_reward = float(np.asarray(rewards).reshape(-1)[0])
            self.current_rewards.append(step_reward)

        # 从 info 中读取通信与能耗指标
        if "energy_efficiency" in info:
            self.current_ee.append(
                float(info["energy_efficiency"])
            )

        if "min_rate_mbps" in info:
            self.current_min_rates.append(
                float(info["min_rate_mbps"])
            )

        if "avg_rate_mbps" in info:
            self.current_avg_rates.append(
                float(info["avg_rate_mbps"])
            )

        if "total_power" in info:
            self.current_power.append(
                float(info["total_power"])
            )

        if "jain_index" in info:
            self.current_jain.append(
                float(info["jain_index"])
            )

        if "actual_displacement" in info:
            self.current_displacements.append(
                float(info["actual_displacement"])
            )

        if "qos_violation_ratio" in info:
            self.current_qos_violation_ratios.append(
                float(info["qos_violation_ratio"])
            )

        if "tracking_distance" in info:
            self.current_episode_tracking_distances.append(
                float(info["tracking_distance"])
            )

        # DummyVecEnv 通常会在 episode 结束时提供 done
        dones = self.locals.get("dones")

        if dones is not None:
            episode_done = bool(
                np.asarray(dones).reshape(-1)[0]
            )

            if episode_done:
                self._finish_episode()

        return True

    def _finish_episode(self):
        """
        汇总当前 episode 数据，并清空临时缓存。
        """

        if len(self.current_rewards) == 0:
            return

        self.episode_mobilities.append(
            self.current_episode_mobility
        )

        # 每个 episode 的累计 reward
        episode_reward = float(
            np.sum(self.current_rewards)
        )

        # 每个 episode 的平均 EE
        episode_mean_ee = (
            float(np.mean(self.current_ee))
            if self.current_ee
            else np.nan
        )

        # episode 内每一步最差用户速率的平均值
        episode_min_rate = (
            float(np.mean(self.current_min_rates))
            if self.current_min_rates
            else np.nan
        )

        episode_avg_rate = (
            float(np.mean(self.current_avg_rates))
            if self.current_avg_rates
            else np.nan
        )

        episode_mean_power = (
            float(np.mean(self.current_power))
            if self.current_power
            else np.nan
        )

        episode_mean_jain = (
            float(np.mean(self.current_jain))
            if self.current_jain
            else np.nan
        )

        # 一个 episode 内 UAV 实际移动距离之和
        episode_travel_distance = (
            float(np.sum(self.current_displacements))
            if self.current_displacements
            else 0.0
        )

        episode_qos_violation_ratio = (
            float(
                np.mean(
                    self.current_qos_violation_ratios
                )
            )
            if self.current_qos_violation_ratios
            else np.nan
        )

        episode_mean_los_users = (
            float(
                np.mean(
                    self.current_los_users
                )
            )
            if self.current_los_users
            else np.nan
        )

        episode_mean_nlos_users = (
            float(
                np.mean(
                    self.current_nlos_users
                )
            )
            if self.current_nlos_users
            else np.nan
        )

        self.episode_rewards.append(
            episode_reward
        )

        self.episode_mean_ee.append(
            episode_mean_ee
        )
        episode_mean_tracking_distance = (
            float(
                np.mean(
                    self.current_episode_tracking_distances
                )
            )
            if self.current_episode_tracking_distances
            else np.nan
        )

        self.episode_mean_tracking_distances.append(
            episode_mean_tracking_distance
        )

        self.episode_min_rates.append(
            episode_min_rate
        )

        self.episode_avg_rates.append(
            episode_avg_rate
        )

        self.episode_mean_power.append(
            episode_mean_power
        )

        self.episode_jain_indices.append(
            episode_mean_jain
        )

        self.episode_travel_distances.append(
            episode_travel_distance
        )

        self.episode_qos_violation_ratios.append(
            episode_qos_violation_ratio
        )
        self.episode_mean_los_users.append(
            episode_mean_los_users
        )

        self.episode_mean_nlos_users.append(
            episode_mean_nlos_users
        )

        # ============================================================
        # V5.2 episode stability summary
        # ============================================================

        episode_mean_power_allocation_change = (
            float(
                np.mean(
                    self.current_power_allocation_changes
                )
            )
            if self.current_power_allocation_changes
            else np.nan
        )

        self.episode_mean_power_allocation_changes.append(
            episode_mean_power_allocation_change
        )

        episode_mean_rate_variation = (
            float(
                np.mean(
                    self.current_rate_variations
                )
            )
            if self.current_rate_variations
            else np.nan
        )

        self.episode_mean_rate_variations.append(
            episode_mean_rate_variation
        )

        episode_p95_rate_variation = (
            float(
                np.percentile(
                    self.current_rate_variations,
                    95
                )
            )
            if self.current_rate_variations
            else np.nan
        )

        self.episode_p95_rate_variations.append(
            episode_p95_rate_variation
        )
        # 清空当前 episode 缓存
        self.current_rewards = []
        self.current_ee = []
        self.current_min_rates = []
        self.current_avg_rates = []
        self.current_power = []
        self.current_jain = []
        self.current_displacements = []
        self.current_qos_violation_ratios = []

        self.current_los_users = []
        self.current_nlos_users = []
        self.current_episode_tracking_distances = []

        self.current_power_allocation_changes = []
        self.current_rate_variations = []
        self.previous_user_rates = None
        self.current_episode_mobility = None

    def _on_training_end(self) -> None:
        """
        不保存训练结束时未完成的 episode，
        避免最终统计被短 episode 污染。
        """
        if self.current_rewards and self.verbose > 0:
            print(
                "Discarding incomplete final episode with "
                f"{len(self.current_rewards)} steps."
            )

    def save_to_file(
        self,
        filename="training_data.npy"
    ):
        """
        保存为 numpy 字典文件，供 monitor_training.py 绘图。
        """

        data = {
            "episode_rewards": np.asarray(
                self.episode_rewards,
                dtype=np.float32
            ),
            "episode_mobilities": np.asarray(
                self.episode_mobilities,
                dtype="<U16"
            ),
            "episode_mean_ee": np.asarray(
                self.episode_mean_ee,
                dtype=np.float64
            ),
            "episode_min_rates": np.asarray(
                self.episode_min_rates,
                dtype=np.float32
            ),
            "episode_mean_tracking_distances":
                np.asarray(
                    self.episode_mean_tracking_distances,
                    dtype=np.float32
                ),
            "episode_avg_rates": np.asarray(
                self.episode_avg_rates,
                dtype=np.float32
            ),
            "episode_mean_power": np.asarray(
                self.episode_mean_power,
                dtype=np.float32
            ),
            "episode_jain_indices": np.asarray(
                self.episode_jain_indices,
                dtype=np.float32
            ),
            "episode_travel_distances": np.asarray(
                self.episode_travel_distances,
                dtype=np.float32
            ),
            "episode_qos_violation_ratios": np.asarray(
                self.episode_qos_violation_ratios,
                dtype=np.float32
            ),
            "episode_mean_los_users": np.asarray(
                self.episode_mean_los_users,
                dtype=np.float32
            ),
            "episode_mean_power_allocation_changes":
                np.asarray(
                    self.episode_mean_power_allocation_changes,
                    dtype=np.float32
                ),

            "episode_mean_rate_variations":
                np.asarray(
                    self.episode_mean_rate_variations,
                    dtype=np.float32
                ),

            "episode_p95_rate_variations":
                np.asarray(
                    self.episode_p95_rate_variations,
                    dtype=np.float32
                ),
            "episode_mean_nlos_users": np.asarray(
                self.episode_mean_nlos_users,
                dtype=np.float32
            )
        }

        np.save(
            filename,
            data,
            allow_pickle=True
        )

        print(
            f"✅ Training summary data saved to: "
            f"{filename}"
        )
def train_general_model(
    steps=400000,
    experiment_name="uav_v5_6_mixed",
    scenario="mixed",
    use_interference=False,
    interference_coupling=0.0,
    power_allocation_mode="learned",
    seed=42
):
    np.random.seed(seed)
    # 1. 设置路径与日志
    log_dir = f"./ppo_logs/{experiment_name}/"
    model_save_path = f"./models/{experiment_name}/model"
    result_dir = f"./results/{experiment_name}"
    data_save_path = f"{result_dir}/training_data.npy"
    config_save_path = f"{result_dir}/training_config.json"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 2. 初始化环境 (强制设置为 "train" 模式)
    num_cpu = 1
    # 注意：确保 'UAV-v0' 在你的环境中已正确注册，否则请直接使用 class 实例化
    env = DummyVecEnv([
        lambda: UAV6GMultiUserEnv(
            num_users=5,
            mode="train",
            mobility=scenario,
            use_interference=use_interference,
            interference_coupling=interference_coupling,
            power_allocation_mode=(
                power_allocation_mode
            )
        )
    ])


    # Ensure the environment starts from the same
    # reproducible random state for this training seed.
    env.seed(seed)
    # ============================================================
    # Runtime configuration verification
    # ============================================================
    import inspect

    base_env = env.envs[0]

    print("=" * 70)
    print("Runtime environment verification")
    print("=" * 70)

    print(
        "Imported environment file:",
        inspect.getfile(UAV6GMultiUserEnv)
    )

    print(
        "Experiment name:",
        experiment_name
    )

    print(
        "use_interference argument:",
        use_interference
    )

    print(
        "interference_coupling argument:",
        interference_coupling
    )

    print(
        "Environment use_interference:",
        base_env.use_interference
    )

    print(
        "Environment interference_coupling:",
        base_env.interference_coupling
    )
    print(
        "Power allocation mode:",
        base_env.power_allocation_mode
    )

    print(
        "Configured mobility:",
        base_env.mobility
    )

    print(
        "Initial active mobility:",
        base_env.active_mobility
    )

    print(
        "Evacuation probability:",
        base_env.evacuation_probability
    )

    print(
        "Environment mode:",
        base_env.mode
    )

    print("=" * 70)

    training_config = {
            "experiment_name": experiment_name,
            "environment_version": "V5.6",
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "total_timesteps": int(steps),
            "training_mobility": scenario,
            "mixed_mobility_probabilities": {
                "evacuation": float(
                    base_env.evacuation_probability
                ),
                "straggler": float(
                    1.0 - base_env.evacuation_probability
                )
            } if scenario == "mixed" else None,
            "use_interference": bool(
                use_interference
            ),
            "interference_coupling": float(
                interference_coupling
            ),
            "power_allocation_mode":
                power_allocation_mode,
            "seed": int(seed),
            "num_users": 5,
            "episode_length": int(
                base_env.MAX_STEPS
            ),
            "learning_rate": 3e-4,
            "entropy_coefficient": 0.01,
            "device": "cpu"
        }
    
    with open(
        config_save_path,
        "w",
        encoding="utf-8"
    ) as config_file:
        json.dump(
            training_config,
            config_file,
            indent=4
        )

    print(
        f"Training config saved before training: "
        f"{config_save_path}"
    )
    # 3. 初始化 PPO 模型
    # device="cuda" 强制在 GPU 上训练
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=log_dir, 
        device="cpu",
        seed=seed
    )

    
    print(f"🚀 开始训练通用模式（General Mode）...")
    print(f"日志将记录到: {log_dir}")
    
    # 4. 开始训练
    monitor_cb = TrainingMonitorCallback()

    model.learn(
        total_timesteps=steps,
        tb_log_name=experiment_name,
        callback=monitor_cb
    )

    # 5. 检查实际 episode 场景分布
    mobility_array = np.asarray(
        monitor_cb.episode_mobilities
    )

    unique_mobilities, mobility_counts = np.unique(
        mobility_array,
        return_counts=True
    )

    print("=" * 70)
    print("Training episode mobility distribution")

    for mobility_name, mobility_count in zip(
        unique_mobilities,
        mobility_counts
    ):
        print(
            f"{mobility_name}: "
            f"{mobility_count} episodes"
        )

    print("=" * 70)

    # 6. 保存训练后的模型和 episode 指标
    model.save(model_save_path)
    monitor_cb.save_to_file(data_save_path)

    # 7. 绘制训练曲线
    plot_training_metrics(data_save_path)

    print(
        f"Model saved to: {model_save_path}.zip"
    )
    print(
        f"Training data saved to: {data_save_path}"
    )
    print(
        f"Training config saved to: {config_save_path}"
    )

# if __name__ == "__main__":

#     # train_general_model(
#     #     steps=1_000_000,
#     #     experiment_name="uav_v5_4_snr_evacuation_seed42",
#     #     mobility="evacuation",#"straggler",
#     #     use_interference=False,
#     #     interference_coupling=0.0,
#     #     seed=42
#     # )
#     train_general_model(
#         steps=1_000_000,
#         experiment_name=(
#             "uav_v5_6_snr_"
#             "mixed50_smooth_1m_seed42"
#         ),
#         scenario="mixed",
#         use_interference=False,
#         interference_coupling=0.1,
#         power_allocation_mode="learned",
#         seed=42
#     )

if __name__ == "__main__":

    TRAIN_SEEDS = [
        0,
        123,
    ]

    for seed in TRAIN_SEEDS:

        experiment_name = (
            "uav_v5_6_sinr_rho01_"
            "mixed50_smooth_1m_"
            f"seed{seed}"
        )

        print("\n" + "=" * 70)
        print(
            f"Training Group A with seed = {seed}"
        )
        print(
            f"Experiment: {experiment_name}"
        )
        print("=" * 70)

        model_path = os.path.join(
            "./models",
            experiment_name,
            "model.zip"
        )

        if os.path.exists(model_path):
            print(
                f"Model already exists: {model_path}"
            )
            print(
                f"Skipping seed {seed}."
            )
            continue

        train_general_model(
            steps=1_000_000,
            experiment_name=experiment_name,
            scenario="mixed",
            use_interference=True,
            interference_coupling=0.1,
            power_allocation_mode="learned",
            seed=seed
        )

#nohup python train.py > train_v5_5_sinr_rho01_mixed50_1m_seed42.log 2>&1 &
#nohup python train.py > train_v5_3_0716.log 2>&1 &