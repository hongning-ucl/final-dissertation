import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(
    values,
    window_size=5
):
    """
    简单移动平均。

    如果数据数量少于窗口大小，直接返回原数据。
    """
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) < window_size:
        return values

    kernel = np.ones(
        window_size,
        dtype=np.float64
    ) / window_size

    smoothed = np.convolve(
        values,
        kernel,
        mode="valid"
    )

    return smoothed


def plot_raw_and_smoothed(
    ax,
    values,
    title,
    ylabel,
    smoothing_window=5,
    scale=1.0
):
    """
    在同一张子图中绘制原始曲线和移动平均曲线。
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values / scale

    episodes = np.arange(
        1,
        len(values) + 1
    )

    ax.plot(
        episodes,
        values,
        alpha=0.30,
        linewidth=1.0,
        label="Raw"
    )

    if (
        smoothing_window > 1
        and len(values) >= smoothing_window
    ):
        smoothed = moving_average(
            values,
            window_size=smoothing_window
        )

        smooth_episodes = np.arange(
            smoothing_window,
            len(values) + 1
        )

        ax.plot(
            smooth_episodes,
            smoothed,
            linewidth=2.0,
            label=(
                f"Moving average "
                f"(window={smoothing_window})"
            )
        )

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


def plot_training_metrics(
    filename="training_data.npy",
    output_filename="training_summary.png",
    smoothing_window=5
):
    """
    生成 3×2 训练总结图。

    六个指标：
    1. Episode Reward
    2. Mean Energy Efficiency
    3. Minimum User Rate
    4. Mean Total Power
    5. Jain Fairness Index
    6. UAV Travel Distance
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"找不到训练数据文件：{filename}"
        )

    loaded = np.load(
        filename,
        allow_pickle=True
    )

    # np.save(dict) 读取后通常是 0 维 object array
    if isinstance(loaded, np.ndarray):
        data = loaded.item()
    else:
        data = loaded

    required_keys = [
        "episode_rewards",
        "episode_mean_ee",
        "episode_min_rates",
        "episode_avg_rates",
        "episode_mean_power",
        "episode_jain_indices",
        "episode_travel_distances",
        "episode_qos_violation_ratios",
        "episode_mean_los_users",
        "episode_mean_nlos_users",
        "episode_mean_tracking_distances"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            "training_data.npy 缺少字段："
            + ", ".join(missing_keys)
            + "\n请确认已经使用新版 "
              "TrainingMonitorCallback 重新训练。"
        )

    rewards = np.asarray(
        data["episode_rewards"]
    )

    mean_ee = np.asarray(
        data["episode_mean_ee"]
    )

    min_rates = np.asarray(
        data["episode_min_rates"]
    )

    avg_rates = np.asarray(
        data["episode_avg_rates"]
    )

    mean_power = np.asarray(
        data["episode_mean_power"]
    )

    jain_indices = np.asarray(
        data["episode_jain_indices"]
    )

    travel_distances = np.asarray(
        data["episode_travel_distances"]
    )

    qos_violation_ratios = np.asarray(
        data["episode_qos_violation_ratios"]
    )

    mean_los_users = np.asarray(
        data["episode_mean_los_users"],
        dtype=np.float64
    )

    mean_nlos_users = np.asarray(
        data["episode_mean_nlos_users"],
        dtype=np.float64
    )

    mean_tracking_distances = np.asarray(
        data["episode_mean_tracking_distances"],
        dtype=np.float64
    )

    episode_count = len(rewards)

    if episode_count == 0:
        raise ValueError(
            "训练数据中没有完整 episode。"
        )

    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(
        f"Number of episodes: "
        f"{episode_count}"
    )
    print(
        f"Final episode reward: "
        f"{rewards[-1]:.4f}"
    )
    print(
        f"Final mean EE: "
        f"{mean_ee[-1] / 1e6:.4f} Mbit/J"
    )
    print(
        f"Final minimum rate: "
        f"{min_rates[-1]:.4f} Mbps"
    )
    print(
        f"Final average rate: "
        f"{avg_rates[-1]:.4f} Mbps"
    )
    print(
        f"Final mean power: "
        f"{mean_power[-1]:.4f} W"
    )
    print(
        f"Final Jain index: "
        f"{jain_indices[-1]:.4f}"
    )
    print(
        f"Final travel distance: "
        f"{travel_distances[-1]:.4f} m"
    )
    print(
        f"Final QoS violation ratio: "
        f"{qos_violation_ratios[-1]:.4f}"
    )

    print(
        f"Final mean LoS users: "
        f"{mean_los_users[-1]:.4f}"
    )

    print(
        f"Final mean NLoS users: "
        f"{mean_nlos_users[-1]:.4f}"
    )
    print(
        f"Final mean tracking distance: "
        f"{mean_tracking_distances[-1]:.4f} m"
    )


    fig = plt.figure(
        figsize=(18, 14)
    )

    # ============================================================
    # 1. Episode Reward
    # ============================================================
    ax1 = fig.add_subplot(
        3,
        2,
        1
    )

    plot_raw_and_smoothed(
        ax=ax1,
        values=rewards,
        title="Training Reward",
        ylabel="Episode Return",
        smoothing_window=smoothing_window
    )

    # ============================================================
    # 2. Energy Efficiency
    # ============================================================
    ax2 = fig.add_subplot(
        3,
        2,
        2
    )

    plot_raw_and_smoothed(
        ax=ax2,
        values=mean_ee,
        title="Mean Energy Efficiency",
        ylabel="Energy Efficiency (Mbit/J)",
        smoothing_window=smoothing_window,
        scale=1e6
    )

    # ============================================================
    # 3. Minimum and Average User Rate
    # ============================================================
    ax3 = fig.add_subplot(
        3,
        2,
        3
    )

    episodes = np.arange(
        1,
        len(min_rates) + 1
    )

    ax3.plot(
        episodes,
        min_rates,
        alpha=0.35,
        linewidth=1.0,
        label="Minimum rate (raw)"
    )

    ax3.plot(
        episodes,
        avg_rates,
        alpha=0.35,
        linewidth=1.0,
        label="Average rate (raw)"
    )

    if len(min_rates) >= smoothing_window:
        smooth_episodes = np.arange(
            smoothing_window,
            len(min_rates) + 1
        )

        ax3.plot(
            smooth_episodes,
            moving_average(
                min_rates,
                smoothing_window
            ),
            linewidth=2.0,
            label="Minimum rate (smoothed)"
        )

        ax3.plot(
            smooth_episodes,
            moving_average(
                avg_rates,
                smoothing_window
            ),
            linewidth=2.0,
            label="Average rate (smoothed)"
        )

    ax3.set_title(
        "User Communication Performance"
    )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Rate (Mbps)")
    ax3.grid(True)
    ax3.legend()

    # ============================================================
    # 4. Mean Total Power
    # ============================================================
    ax4 = fig.add_subplot(
        3,
        2,
        4
    )

    plot_raw_and_smoothed(
        ax=ax4,
        values=mean_power,
        title="Mean Total Power Consumption",
        ylabel="Power (W)",
        smoothing_window=smoothing_window
    )

    # ============================================================
    # 5. Jain Fairness Index
    # ============================================================
    ax5 = fig.add_subplot(
        3,
        2,
        5
    )

    plot_raw_and_smoothed(
        ax=ax5,
        values=jain_indices,
        title="Jain Fairness Index",
        ylabel="Jain Index",
        smoothing_window=smoothing_window
    )

    ax5.set_ylim(
        0.0,
        1.05
    )

    # ============================================================
    # 6. UAV Travel Distance
    # ============================================================
    ax6 = fig.add_subplot(
        3,
        2,
        6
    )

    plot_raw_and_smoothed(
        ax=ax6,
        values=travel_distances,
        title="UAV Travel Distance",
        ylabel="Travel Distance per Episode (m)",
        smoothing_window=smoothing_window
    )

    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("=" * 60)
    print(
        f"✅ Training summary figure saved to: "
        f"{output_filename}"
    )
    print("=" * 60)

    # ============================================================
    # 单独保存 QoS violation 曲线
    # ============================================================
    qos_output_filename = (
        "training_qos_violation.png"
    )

    fig_qos = plt.figure(
        figsize=(10, 6)
    )

    ax_qos = fig_qos.add_subplot(
        1,
        1,
        1
    )

    plot_raw_and_smoothed(
        ax=ax_qos,
        values=qos_violation_ratios,
        title="QoS Violation Ratio",
        ylabel="Violation Ratio",
        smoothing_window=smoothing_window
    )

    ax_qos.set_ylim(
        0.0,
        1.05
    )

    plt.tight_layout()

    plt.savefig(
        qos_output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_qos)

    print(
        f"✅ QoS violation figure saved to: "
        f"{qos_output_filename}"
    )

        # ============================================================
    # 单独保存 LoS / NLoS 用户数量曲线
    # ============================================================
    los_output_filename = (
        "training_los_nlos_users.png"
    )

    fig_los = plt.figure(
        figsize=(10, 6)
    )

    ax_los = fig_los.add_subplot(
        1,
        1,
        1
    )

    episodes_los = np.arange(
        1,
        len(mean_los_users) + 1
    )

    # 原始 LoS 曲线
    ax_los.plot(
        episodes_los,
        mean_los_users,
        alpha=0.30,
        linewidth=1.0,
        label="Mean LoS users (raw)"
    )

    # 原始 NLoS 曲线
    ax_los.plot(
        episodes_los,
        mean_nlos_users,
        alpha=0.30,
        linewidth=1.0,
        label="Mean NLoS users (raw)"
    )

    # 移动平均曲线
    if (
        smoothing_window > 1
        and len(mean_los_users) >= smoothing_window
    ):
        smooth_episodes_los = np.arange(
            smoothing_window,
            len(mean_los_users) + 1
        )

        ax_los.plot(
            smooth_episodes_los,
            moving_average(
                mean_los_users,
                smoothing_window
            ),
            linewidth=2.0,
            label=(
                "Mean LoS users "
                f"(moving average={smoothing_window})"
            )
        )

        ax_los.plot(
            smooth_episodes_los,
            moving_average(
                mean_nlos_users,
                smoothing_window
            ),
            linewidth=2.0,
            label=(
                "Mean NLoS users "
                f"(moving average={smoothing_window})"
            )
        )

    ax_los.set_title(
        "LoS and NLoS User Count During Training"
    )

    ax_los.set_xlabel(
        "Episode"
    )

    ax_los.set_ylabel(
        "Mean Number of Users per Episode"
    )

    ax_los.set_ylim(
        0.0,
        5.2
    )

    ax_los.set_yticks(
        np.arange(
            0,
            6,
            1
        )
    )

    ax_los.grid(
        True
    )

    ax_los.legend()

    plt.tight_layout()

    plt.savefig(
        los_output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig_los
    )

    print(
        f"✅ LoS/NLoS figure saved to: "
        f"{los_output_filename}"
    )
        # ============================================================
    # 单独保存 Tracking Distance 曲线
    # ============================================================
    tracking_output_filename = (
        "training_tracking_distance.png"
    )

    fig_tracking = plt.figure(
        figsize=(10, 6)
    )

    ax_tracking = fig_tracking.add_subplot(
        1,
        1,
        1
    )

    plot_raw_and_smoothed(
        ax=ax_tracking,
        values=mean_tracking_distances,
        title="Mean UAV-to-User-Centroid Tracking Distance",
        ylabel="Tracking Distance (m)",
        smoothing_window=smoothing_window
    )

    plt.tight_layout()

    plt.savefig(
        tracking_output_filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig_tracking
    )

    print(
        f"✅ Tracking-distance figure saved to: "
        f"{tracking_output_filename}"
    )
if __name__ == "__main__":
    plot_training_metrics(
        filename="training_data.npy",
        output_filename="training_summary.png",
        smoothing_window=5
    )