import gymnasium as gym
from gymnasium import spaces
import numpy as np


class UAV6GMultiUserEnv(gym.Env):
    """
    Multi-user UAV communication environment with joint
    trajectory and power control.

    V5.4:
    1. Fixed 5-user setting with 48-dimensional observation.
    2. Joint UAV trajectory, total transmit power and
    per-user power-allocation control.
    3. Optional residual-interference SINR model.
    4. Supported mobility scenarios:
    "static", "evacuation", and "straggler".

    """

    def __init__(
        self,
        num_users=5,
        mode="train",
        scenario="static",
        mobility=None,
        use_interference=False,  #是否启用干扰模型
        interference_coupling=0.0, #干扰强度
        power_allocation_mode="learned",
        building_layout= "default",
    ):
        super(UAV6GMultiUserEnv, self).__init__()
  
        self.MAX_USERS_LIMIT = 10
        assert num_users == 5, (
            "当前第一阶段环境固定支持5个用户"
        )
        self.num_users = num_users
        self.mode = mode.lower()  
        self.scenario = scenario.lower()
        self.mobility = mobility
        self.building_layout = (
            str(building_layout)
            .lower()
            .strip()
        )

        self.SUPPORTED_BUILDING_LAYOUTS = {
            "default",
            "dense",
        }

        if (
            self.building_layout
            not in self.SUPPORTED_BUILDING_LAYOUTS
        ):
            raise ValueError(
                f"Unsupported building layout: "
                f"{self.building_layout}. "
                f"Supported layouts are: "
                f"{sorted(self.SUPPORTED_BUILDING_LAYOUTS)}"
            )

        # ------------------------------
        # Backward compatibility
        # ------------------------------
        if self.mobility is None:
            self.mobility = self.scenario

        # ============================================================
        # V5.4 Scenario validation
        # ============================================================
        self.SUPPORTED_MOBILITIES = {
            "static",
            "evacuation",
            "straggler",
            "random_walk",
            "mixed",
        }
        self.mobility = self.mobility.lower().strip()
        if self.mobility not in self.SUPPORTED_MOBILITIES:
            raise ValueError(
                f"Unsupported mobility: {self.mobility}. "
                f"Supported mobilities are: "
                f"{sorted(self.SUPPORTED_MOBILITIES)}"
            )

        # ============================================================
        # V5.4 Interference configuration
        # ============================================================

        # 是否启用干扰模型
        self.use_interference = bool(use_interference)

        # 干扰耦合系数 rho
        self.interference_coupling = float(interference_coupling)

        self.power_allocation_mode = (
            str(power_allocation_mode)
            .lower()
            .strip()
        )

        supported_power_allocation_modes = {
            "learned",
            "equal",
        }

        if (
            self.power_allocation_mode
            not in supported_power_allocation_modes
        ):
            raise ValueError(
                "Unsupported power_allocation_mode: "
                f"{self.power_allocation_mode}. "
                "Supported modes are: "
                f"{sorted(supported_power_allocation_modes)}"
            )        

        # 防止输入非法值
        if not 0.0 <= self.interference_coupling <= 1.0:
            raise ValueError(
                "interference_coupling must be between 0 and 1."
            )
        
        self.MAX_STEPS = 200
        self.current_step = 0
        self.dt = 1.0  
        self.INITIAL_BATTERY = 200.0
        
        self.MAP_LIMIT = 500.0
        self.MIN_ALTITUDE = 20.0   
        self.MAX_ALTITUDE = 100.0
   
        
        # 14维动作空间 (4维移动及睡眠控制 + 10维全量发射功率偏好)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4 + self.MAX_USERS_LIMIT,), dtype=np.float32)
        # 48维状态空间：
        # 4维 UAV 状态
        # + 20维用户相对位置
        # + 14维上一时刻动作
        # + 5维用户归一化速率
        # + 5维用户 LoS/NLoS 状态
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )
        # 无线信道物理常量
        self.f_c = 2.5e9       # 2.5 GHz
        self.B = 20.0e6        # 20 MHz 带宽
        self.sigma2 = 1.0e-12  # 噪声功率 -90 dBm
        self.P_max = 0.1       # 最大下行发射功率 100 mW (0.1 W)
        # V4.3: 可变总发射功率范围
        self.P_min = 0.01

        # ============================================================
        # V5.5 Mixed-mobility training configuration
        # ============================================================

        # mixed模式下，evacuation出现的概率。
        # 0.5表示evacuation和straggler各占50%。
        self.evacuation_probability = 0.5

        # 当前episode真正采用的mobility。
        self.active_mobility = self.mobility

        # ============================================================
        # V5.2 Stable resource-control parameters
        # ============================================================

        # 当前功率分配中，采用多少比例的 PPO 新决策。
        # beta 越小，功率分配越平滑，但响应弱用户变化越慢。
        self.POWER_ALLOCATION_SMOOTHING_BETA = 0.20

        # 功率分配变化惩罚权重。
        # 使用实际功率比例之间的 total-variation distance。
        self.POWER_ALLOCATION_CHANGE_WEIGHT = 0.05

        # ============================================================
        # V5.6 UAV movement smoothing parameters
        # ============================================================

        # 一阶低通滤波系数：
        # 越小越平滑，但 UAV 对用户移动的响应越慢。
        # 推荐先使用 0.25。
        self.MOVEMENT_SMOOTHING_BETA = 0.25

        # 小动作死区，作用于平滑后的动作。
        # 只有非常微小的控制量才被置零。
        self.MOVEMENT_DEAD_ZONE = 0.03

        # 方向反转惩罚权重
        self.DIRECTION_REVERSAL_WEIGHT = 0.10
        # ============================================================
        # V5.6 Adaptive altitude-hold parameters
        # ============================================================

        # 只有 UAV 达到该高度后，才允许进入高度保持状态。
        # 避免影响前期从 30 m 向上爬升。
        self.ALTITUDE_HOLD_MIN_ALTITUDE = 85.0

        # 高度保持带宽：目标高度上下各允许 2 m。
        self.ALTITUDE_HOLD_BAND = 2.0

        # 当垂直动作绝对值低于该值时，
        # 将其视为 PPO 的微小调整，而不是明确升降指令。
        self.ALTITUDE_HOLD_ACTION_THRESHOLD = 0.20

        # 当 PPO 给出较强的升降动作时，解除高度保持。
        self.ALTITUDE_HOLD_RELEASE_THRESHOLD = 0.40
        # ============================================================
        # V4.1: Urban building and LoS/NLoS parameters
        # ============================================================

        # LoS 与 NLoS 额外路径损耗
        self.LOS_EXTRA_LOSS_DB = 1.0
        self.NLOS_EXTRA_LOSS_DB = 20.0

        # 建筑物定义：
        # x_min, x_max, y_min, y_max 为水平边界
        # height 为建筑高度
        self.buildings = (
            self._create_building_layout(
                self.building_layout
            )
        )
        
        # 3D 建筑物障碍物实体定义
        # self.buildings = [
        #     {'x_min': 150.0, 'x_max': 250.0, 'y_min': 150.0, 'y_max': 250.0, 'height': 45.0},  # 大楼 1
        #     {'x_min': 300.0, 'x_max': 450.0, 'y_min': 250.0, 'y_max': 400.0, 'height': 35.0}   # 大楼 2
        # ]
        
        self.user_rates_history = np.zeros(self.num_users)
        self.last_pos = np.zeros(3, dtype=np.float32) # 添加这行，初始化为零向量
        self.current_rates = np.zeros(self.num_users) # 初始化
        self.last_action = np.zeros(4 + self.MAX_USERS_LIMIT, dtype=np.float32)

        # V5.6 上一时刻实际执行的平滑移动动作
        self.previous_movement_action = np.zeros(
            3,
            dtype=np.float32
        )
        # 当前高度保持目标。
        # None 表示尚未进入高度保持状态。
        self.altitude_hold_target = None

        self.reset()
    def _create_building_layout(
        self,
        layout_name
    ):
        """
        Create a predefined urban building layout.

        default:
            Original two-building environment used during training.

        dense:
            Unseen four-building environment used only for
            generalisation evaluation.
        """

        if layout_name == "default":

            return [
                {
                    "name": "Building 1",
                    "x_min": 180.0,
                    "x_max": 260.0,
                    "y_min": 150.0,
                    "y_max": 280.0,
                    "height": 45.0,
                },
                {
                    "name": "Building 2",
                    "x_min": 300.0,
                    "x_max": 380.0,
                    "y_min": 260.0,
                    "y_max": 380.0,
                    "height": 40.0,
                },
            ]

        elif layout_name == "dense":

            return [
                {
                    "name": "Building 1",
                    "x_min": 180.0,
                    "x_max": 260.0,
                    "y_min": 150.0,
                    "y_max": 280.0,
                    "height": 45.0,
                },
                {
                    "name": "Building 2",
                    "x_min": 300.0,
                    "x_max": 380.0,
                    "y_min": 260.0,
                    "y_max": 380.0,
                    "height": 40.0,
                },
                {
                    "name": "Building 3",
                    "x_min": 60.0,
                    "x_max": 130.0,
                    "y_min": 220.0,
                    "y_max": 320.0,
                    "height": 35.0,
                },
                {
                    "name": "Building 4",
                    "x_min": 390.0,
                    "x_max": 460.0,
                    "y_min": 300.0,
                    "y_max": 390.0,
                    "height": 50.0,
                },
            ]

        raise ValueError(
            f"Unknown building layout: "
            f"{layout_name}"
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ============================================================
        # V5.5 Select mobility at episode reset
        # ============================================================

        if self.mobility == "mixed":
            if self.np_random.random() < self.evacuation_probability:
                self.active_mobility = "evacuation"
            else:
                self.active_mobility = "straggler"
        else:
            self.active_mobility = self.mobility

        self.current_step = 0
        self.battery = self.INITIAL_BATTERY

        self.uav_pos = np.array(
            [50.0, 50.0, 30.0],
            dtype=np.float32
        )

        self.last_pos = self.uav_pos.copy()

        self.last_action = np.zeros(
                    4 + self.MAX_USERS_LIMIT,
                    dtype=np.float32
                )
        self.previous_movement_action = np.zeros(
            3,
            dtype=np.float32
        )

        self.altitude_hold_target = None
        # ============================================================
        # V5.2 Power-allocation history
        # ============================================================

        # Dynamic-pool softmax weights。
        # 初始化为均匀分配。
        self.previous_power_weights = np.full(
            self.num_users,
            1.0 / self.num_users,
            dtype=np.float32
        )

        # 实际总功率中每个用户获得的比例。
        # 包含 floor power 和 dynamic power。
        self.previous_power_allocation_fractions = np.full(
            self.num_users,
            1.0 / self.num_users,
            dtype=np.float32
        )
        # 第一阶段固定用户拓扑
        self.user_positions = np.array([
            [150.0, 150.0],
            [350.0, 100.0],
            [400.0, 200.0],
            [100.0, 400.0],
            [300.0, 420.0]
        ], dtype=np.float32)

        if self.active_mobility == "evacuation":
            #self._update_evacuation_users()

            # ============================================================
            # V5.1 Evacuation
            # ============================================================

            self.evacuation_target = np.array(
                [480.0, 480.0],
                dtype=np.float32
            )

            self.user_speeds = np.array(
                [0.40, 0.425, 0.38, 0.415, 0.39],
                dtype=np.float32
            )

            direction_vectors = (
                self.evacuation_target[None, :]
                - self.user_positions
            )

            direction_norms = np.linalg.norm(
                direction_vectors,
                axis=1,
                keepdims=True
            )

            direction_norms = np.maximum(
                direction_norms,
                1e-6
            )

            unit_directions = (
                direction_vectors
                / direction_norms
            )

            self.user_velocities = (
                unit_directions
                * self.user_speeds[:, None]
            ).astype(np.float32)

        elif self.active_mobility == "straggler":
           # self._update_straggler_users()
           # ============================================================
            # V5.4 Straggler mobility initialization
            # ============================================================

            # straggler 不使用 evacuation target
            self.evacuation_target = None

            # User 1–4 的共同活动中心
            self.straggler_group_center = np.array(
                [250.0, 250.0],
                dtype=np.float64
            )

            # User 1–4 最多距离群体中心 150 m。
            # 该值需要覆盖当前四名用户的初始位置。
            self.straggler_group_radius = 150.0

            # User 1–4 的移动速度
            self.straggler_group_speeds = np.array(
                [0.40, 0.425, 0.38, 0.415],
                dtype=np.float32
            )

            # User 1–4 的初始运动方向
            self.straggler_headings = self.np_random.uniform(
                low=-np.pi,
                high=np.pi,
                size=4
            ).astype(np.float64)

            # User 5 的大范围移动速度
            self.straggler_roaming_speed = 1.20

            # User 5 的第一个 waypoint
            self.straggler_waypoint = np.array(
                [60.0, 80.0],
                dtype=np.float64
            )

            self.user_speeds = np.array(
                [
                    self.straggler_group_speeds[0],
                    self.straggler_group_speeds[1],
                    self.straggler_group_speeds[2],
                    self.straggler_group_speeds[3],
                    self.straggler_roaming_speed
                ],
                dtype=np.float32
            )

            self.user_velocities = np.zeros(
                (self.num_users, 2),
                dtype=np.float32
        )
        elif self.active_mobility == "random_walk":

            # Random walk does not use a common target.
            self.evacuation_target = None

            # Different but comparable user speeds.
            self.random_walk_speeds = np.array(
                [
                    0.40,
                    0.425,
                    0.38,
                    0.415,
                    0.39,
                ],
                dtype=np.float32,
            )

            # Each user receives an independent initial heading.
            self.random_walk_headings = (
                self.np_random.uniform(
                    low=-np.pi,
                    high=np.pi,
                    size=self.num_users,
                ).astype(np.float64)
            )

            self.user_speeds = (
                self.random_walk_speeds.copy()
            )

            self.user_velocities = np.zeros(
                (self.num_users, 2),
                dtype=np.float32,
            )
        else:

            self.evacuation_target = None

            self.user_speeds = np.zeros(
                self.num_users,
                dtype=np.float32
            )

            self.user_velocities = np.zeros(
                (self.num_users, 2),
                dtype=np.float32
            )
        # ============================================================
        # V5.1 Tracking state
        # ============================================================
        initial_centroid = np.mean(
            self.user_positions,
            axis=0
        )

        self.previous_tracking_distance = float(
            np.linalg.norm(
                self.uav_pos[:2]
                - initial_centroid
            )
        )

        self.current_rates = np.zeros(
            self.num_users,
            dtype=np.float32
        )
        self.current_los_flags = np.zeros(
            self.num_users,
            dtype=np.float32
        )

        self.hunger_weights = np.ones(
            self.num_users,
            dtype=np.float32
        )

        return self._get_obs(), {}
    def _update_evacuation_users(self):
        """
        Update users under the evacuation mobility pattern.

        All users move toward self.evacuation_target.
        Each user uses its own speed and avoids buildings by
        trying several steering directions.
        """

        previous_user_positions = self.user_positions.copy()

        updated_user_positions = self.user_positions.copy()

        actual_user_velocities = np.zeros(
            (self.num_users, 2),
            dtype=np.float32
        )

        # 优先直行；遇到建筑后依次尝试左右转向
        steering_angles_deg = [
            0.0,
            30.0,
            -30.0,
            60.0,
            -60.0,
            90.0,
            -90.0,
            120.0,
            -120.0,
            150.0,
            -150.0,
            180.0
        ]

        for user_idx in range(self.num_users):

            current_position = previous_user_positions[
                user_idx
            ].astype(np.float64)

            target_vector = (
                self.evacuation_target.astype(np.float64)
                - current_position
            )

            remaining_distance = float(
                np.linalg.norm(target_vector)
            )

            user_speed = float(
                self.user_speeds[user_idx]
            )

            # 已到达目标，或该用户速度为 0
            if (
                remaining_distance <= 1e-6
                or user_speed <= 0.0
            ):
                updated_user_positions[user_idx] = (
                    current_position
                )

                actual_user_velocities[user_idx] = 0.0

                continue

            desired_direction = (
                target_vector / remaining_distance
            )

            movement_distance = min(
                user_speed * self.dt,
                remaining_distance
            )

            valid_candidates = []

            for angle_deg in steering_angles_deg:

                angle_rad = np.deg2rad(angle_deg)

                rotation_matrix = np.array(
                    [
                        [
                            np.cos(angle_rad),
                            -np.sin(angle_rad)
                        ],
                        [
                            np.sin(angle_rad),
                            np.cos(angle_rad)
                        ]
                    ],
                    dtype=np.float64
                )

                candidate_direction = (
                    rotation_matrix @ desired_direction
                )

                candidate_position = (
                    current_position
                    + candidate_direction
                    * movement_distance
                )

                candidate_position = np.clip(
                    candidate_position,
                    0.0,
                    self.MAP_LIMIT
                )

                if self.is_user_position_blocked(
                    candidate_position,
                    clearance=1.0
                ):
                    continue

                candidate_distance_to_target = float(
                    np.linalg.norm(
                        self.evacuation_target
                        - candidate_position
                    )
                )

                # 优先靠近目标，其次减少转向角度
                candidate_score = (
                    candidate_distance_to_target
                    + 0.002 * abs(angle_deg)
                )

                valid_candidates.append(
                    (
                        candidate_score,
                        abs(angle_deg),
                        candidate_position
                    )
                )

            if valid_candidates:

                valid_candidates.sort(
                    key=lambda item: (
                        item[0],
                        item[1]
                    )
                )

                selected_position = (
                    valid_candidates[0][2]
                )

            else:

                # 所有候选方向均被阻挡时保持原位
                selected_position = (
                    current_position.copy()
                )

            updated_user_positions[user_idx] = (
                selected_position
            )

            actual_user_velocities[user_idx] = (
                selected_position
                - current_position
            ) / self.dt

        self.user_positions = (
            updated_user_positions.astype(np.float32)
        )

        self.user_velocities = (
            actual_user_velocities.astype(np.float32)
        )

    def _update_straggler_users(self):
        """
        Update users under the straggler mobility pattern.

        Users 1–4:
            Perform persistent local random walks while remaining
            within a bounded region around a shared group centre.

        User 5:
            Moves independently between random waypoints across
            the map, representing a roaming straggler.

        All users avoid buildings and remain inside the map.
        """

        previous_user_positions = self.user_positions.copy()

        updated_user_positions = self.user_positions.copy()

        actual_user_velocities = np.zeros(
            (self.num_users, 2),
            dtype=np.float32
        )

        # ============================================================
        # A. Users 1–4: local bounded random walk
        # ============================================================
        for user_idx in range(4):

            current_position = previous_user_positions[
                user_idx
            ].astype(np.float64)

            current_heading = float(
                self.straggler_headings[user_idx]
            )

            # 每一步只做小幅转向，形成具有持续性的 random walk
            heading_noise = float(
                self.np_random.normal(
                    loc=0.0,
                    scale=np.deg2rad(12.0)
                )
            )

            proposed_heading = (
                current_heading + heading_noise
            )

            current_offset = (
                current_position
                - self.straggler_group_center
            )

            distance_from_group_center = float(
                np.linalg.norm(current_offset)
            )

            # 如果接近或超出群体半径，主动转向群体中心
            if (
                distance_from_group_center
                >= self.straggler_group_radius
            ):

                direction_to_center = (
                    self.straggler_group_center
                    - current_position
                )

                direction_norm = float(
                    np.linalg.norm(direction_to_center)
                )

                if direction_norm > 1e-6:
                    proposed_heading = float(
                        np.arctan2(
                            direction_to_center[1],
                            direction_to_center[0]
                        )
                    )

            user_speed = float(
                self.straggler_group_speeds[user_idx]
            )

            movement_distance = (
                user_speed * self.dt
            )

            # 遇到建筑或边界时尝试其他方向
            steering_offsets_deg = [
                0.0,
                25.0,
                -25.0,
                50.0,
                -50.0,
                90.0,
                -90.0,
                135.0,
                -135.0,
                180.0
            ]

            selected_position = current_position.copy()
            selected_heading = proposed_heading

            for offset_deg in steering_offsets_deg:

                candidate_heading = (
                    proposed_heading
                    + np.deg2rad(offset_deg)
                )

                candidate_direction = np.array(
                    [
                        np.cos(candidate_heading),
                        np.sin(candidate_heading)
                    ],
                    dtype=np.float64
                )

                candidate_position = (
                    current_position
                    + candidate_direction
                    * movement_distance
                )

                candidate_position = np.clip(
                    candidate_position,
                    0.0,
                    self.MAP_LIMIT
                )

                # 不允许进入建筑
                if self.is_user_position_blocked(
                    candidate_position,
                    clearance=1.0
                ):
                    continue


                selected_position = candidate_position
                selected_heading = candidate_heading

                break      
            
            updated_user_positions[user_idx] = (
                selected_position
            )

            actual_user_velocities[user_idx] = (
                selected_position
                - current_position
            ) / self.dt

            self.straggler_headings[user_idx] = (
                selected_heading
            )

        # ============================================================
        # B. User 5: independent waypoint roaming
        # ============================================================
        roaming_user_idx = 4

        current_position = previous_user_positions[
            roaming_user_idx
        ].astype(np.float64)

        waypoint_vector = (
            self.straggler_waypoint
            - current_position
        )

        waypoint_distance = float(
            np.linalg.norm(waypoint_vector)
        )

        # 到达当前 waypoint 后生成一个新的安全 waypoint
        if waypoint_distance <= 5.0:

            new_waypoint_found = False

            for _ in range(100):

                candidate_waypoint = self.np_random.uniform(
                    low=20.0,
                    high=self.MAP_LIMIT - 20.0,
                    size=2
                ).astype(np.float64)

                if self.is_user_position_blocked(
                    candidate_waypoint,
                    clearance=5.0
                ):
                    continue

                # 避免新 waypoint 与当前位置太接近
                if (
                    np.linalg.norm(
                        candidate_waypoint
                        - current_position
                    )
                    < 120.0
                ):
                    continue

                self.straggler_waypoint = (
                    candidate_waypoint
                )

                new_waypoint_found = True
                break

            if not new_waypoint_found:
                self.straggler_waypoint = (
                    current_position.copy()
                )

            waypoint_vector = (
                self.straggler_waypoint
                - current_position
            )

            waypoint_distance = float(
                np.linalg.norm(waypoint_vector)
            )

        if waypoint_distance > 1e-6:

            desired_direction = (
                waypoint_vector / waypoint_distance
            )

            desired_heading = float(
                np.arctan2(
                    desired_direction[1],
                    desired_direction[0]
                )
            )

            movement_distance = min(
                self.straggler_roaming_speed
                * self.dt,
                waypoint_distance
            )

            steering_offsets_deg = [
                0.0,
                20.0,
                -20.0,
                40.0,
                -40.0,
                70.0,
                -70.0,
                100.0,
                -100.0,
                140.0,
                -140.0,
                180.0
            ]

            valid_candidates = []

            for offset_deg in steering_offsets_deg:

                candidate_heading = (
                    desired_heading
                    + np.deg2rad(offset_deg)
                )

                candidate_direction = np.array(
                    [
                        np.cos(candidate_heading),
                        np.sin(candidate_heading)
                    ],
                    dtype=np.float64
                )

                candidate_position = (
                    current_position
                    + candidate_direction
                    * movement_distance
                )

                candidate_position = np.clip(
                    candidate_position,
                    0.0,
                    self.MAP_LIMIT
                )

                if self.is_user_position_blocked(
                    candidate_position,
                    clearance=1.0
                ):
                    continue

                candidate_distance_to_waypoint = float(
                    np.linalg.norm(
                        self.straggler_waypoint
                        - candidate_position
                    )
                )

                candidate_score = (
                    candidate_distance_to_waypoint
                    + 0.002 * abs(offset_deg)
                )

                valid_candidates.append(
                    (
                        candidate_score,
                        abs(offset_deg),
                        candidate_position
                    )
                )

            if valid_candidates:

                valid_candidates.sort(
                    key=lambda item: (
                        item[0],
                        item[1]
                    )
                )

                selected_position = (
                    valid_candidates[0][2]
                )

            else:

                selected_position = (
                    current_position.copy()
                )

        else:

            selected_position = (
                current_position.copy()
            )

        updated_user_positions[
            roaming_user_idx
        ] = selected_position

        actual_user_velocities[
            roaming_user_idx
        ] = (
            selected_position
            - current_position
        ) / self.dt

        # ============================================================
        # C. Save updated states
        # ============================================================
        self.user_positions = (
            updated_user_positions.astype(np.float32)
        )

        self.user_velocities = (
            actual_user_velocities.astype(np.float32)
        )
    def _update_random_walk_users(self):
        """
        Independent persistent random walk for all users.

        Each user moves with a fixed speed while its heading
        changes gradually over time. Users remain within the map
        and avoid building footprints.
        """

        previous_positions = (
            self.user_positions.copy()
        )

        updated_positions = (
            self.user_positions.copy()
        )

        actual_velocities = np.zeros(
            (self.num_users, 2),
            dtype=np.float32,
        )

        steering_offsets_deg = [
            0.0,
            25.0,
            -25.0,
            50.0,
            -50.0,
            90.0,
            -90.0,
            135.0,
            -135.0,
            180.0,
        ]

        for user_idx in range(
            self.num_users
        ):

            current_position = (
                previous_positions[
                    user_idx
                ].astype(np.float64)
            )

            current_heading = float(
                self.random_walk_headings[
                    user_idx
                ]
            )

            # Small random heading change creates
            # a persistent rather than purely random walk.
            heading_noise = float(
                self.np_random.normal(
                    loc=0.0,
                    scale=np.deg2rad(15.0),
                )
            )

            proposed_heading = (
                current_heading
                + heading_noise
            )

            movement_distance = float(
                self.random_walk_speeds[
                    user_idx
                ]
                * self.dt
            )

            selected_position = (
                current_position.copy()
            )

            selected_heading = (
                proposed_heading
            )

            for offset_deg in (
                steering_offsets_deg
            ):

                candidate_heading = (
                    proposed_heading
                    + np.deg2rad(
                        offset_deg
                    )
                )

                direction = np.array(
                    [
                        np.cos(
                            candidate_heading
                        ),
                        np.sin(
                            candidate_heading
                        ),
                    ],
                    dtype=np.float64,
                )

                raw_candidate = (
                    current_position
                    + direction
                    * movement_distance
                )

                # Do not silently clip a movement that
                # crosses the map boundary.
                inside_map = bool(
                    0.0
                    <= raw_candidate[0]
                    <= self.MAP_LIMIT
                    and
                    0.0
                    <= raw_candidate[1]
                    <= self.MAP_LIMIT
                )

                if not inside_map:
                    continue

                if self.is_user_position_blocked(
                    raw_candidate,
                    clearance=1.0,
                ):
                    continue

                selected_position = (
                    raw_candidate
                )

                selected_heading = (
                    candidate_heading
                )

                break

            updated_positions[
                user_idx
            ] = selected_position

            actual_velocities[
                user_idx
            ] = (
                selected_position
                - current_position
            ) / self.dt

            self.random_walk_headings[
                user_idx
            ] = selected_heading

        self.user_positions = (
            updated_positions.astype(
                np.float32
            )
        )

        self.user_velocities = (
            actual_velocities.astype(
                np.float32
            )
        )
    def _get_obs(self):
        """
        构造 48 维 observation：

        1. UAV state: 4
        2. Relative user positions: 20
        3. Previous action: 14
        4. Normalized per-user rates: 5
        5. Per-user LoS flags: 5
        """

        # ============================================================
        # 1. UAV 状态：4维
        # ============================================================
        uav_state = np.array(
            [
                self.uav_pos[0] / self.MAP_LIMIT,
                self.uav_pos[1] / self.MAP_LIMIT,
                self.uav_pos[2] / self.MAX_ALTITUDE,
                self.battery / self.INITIAL_BATTERY
            ],
            dtype=np.float32
        )

        # ============================================================
        # 2. 用户相对位置：10 × 2 = 20维
        # ============================================================
        relative_user_states = np.full(
            (self.MAX_USERS_LIMIT, 2),
            -1.0,
            dtype=np.float32
        )

        for user_idx in range(
            self.num_users
        ):
            relative_user_states[
                user_idx,
                0
            ] = (
                self.user_positions[user_idx, 0]
                - self.uav_pos[0]
            ) / self.MAP_LIMIT

            relative_user_states[
                user_idx,
                1
            ] = (
                self.user_positions[user_idx, 1]
                - self.uav_pos[1]
            ) / self.MAP_LIMIT

        # ============================================================
        # 3. 上一个动作：14维
        # ============================================================
        action_obs = np.asarray(
            self.last_action,
            dtype=np.float32
        ).copy()

        # 前三维使用环境实际执行的平滑移动动作
        action_obs[:3] = (
            self.previous_movement_action
        )

        # ============================================================
        # 4. 每用户归一化速率：5维
        # ============================================================
        # 参考速率设为100 Mbps。
        # 允许观测值最高到2，避免高于100 Mbps的信息完全丢失。
        RATE_OBS_REF = 100.0e6

        normalized_rate_obs = np.clip(
            np.asarray(
                self.current_rates,
                dtype=np.float32
            ) / RATE_OBS_REF,
            0.0,
            2.0
        ).astype(np.float32)

        # ============================================================
        # 5. 每用户 LoS/NLoS 状态：5维
        # ============================================================
        # LoS = 1.0
        # NLoS = 0.0
        los_flag_obs = np.asarray(
            self.current_los_flags,
            dtype=np.float32
        )

        # ============================================================
        # 6. 拼接为48维 observation
        # ============================================================
        obs = np.concatenate(
            [
                uav_state,
                relative_user_states.flatten(),
                action_obs,
                normalized_rate_obs,
                los_flag_obs
            ]
        )

        if obs.shape != (48,):
            raise RuntimeError(
                f"Observation shape mismatch: "
                f"expected (48,), got {obs.shape}"
            )

        if not np.all(
            np.isfinite(obs)
        ):
            raise FloatingPointError(
                "Non-finite observation detected."
            )

        return obs.astype(
            np.float32
        )

    def is_inside_building(self, position):
        """
        判断 UAV 当前三维位置是否进入建筑物实体。

        position:
            [x, y, z]
        """

        x = float(position[0])
        y = float(position[1])
        z = float(position[2])

        for building in self.buildings:
            inside_horizontal_footprint = (
                building["x_min"] <= x <= building["x_max"]
                and building["y_min"] <= y <= building["y_max"]
            )

            below_building_height = (
                z < building["height"]+2.0
            )

            if (
                inside_horizontal_footprint
                and below_building_height
            ):
                return True

        return False

    def is_user_position_blocked(
        self,
        position,
        clearance=1.0
    ):
        """
        判断地面用户的二维位置是否进入建筑物区域。

        Parameters
        ----------
        position:
            用户二维坐标 [x, y]。

        clearance:
            与建筑物边界保留的安全距离，单位为 m。
            默认 1 m，避免用户轨迹紧贴或压在建筑边界上。
        """

        position = np.asarray(
            position,
            dtype=np.float64
        ).reshape(-1)

        x = float(position[0])
        y = float(position[1])

        for building in self.buildings:

            inside_expanded_footprint = (
                building["x_min"] - clearance
                <= x
                <= building["x_max"] + clearance
                and
                building["y_min"] - clearance
                <= y
                <= building["y_max"] + clearance
            )

            if inside_expanded_footprint:
                return True

        return False

    def check_line_of_sight(
        self,
        uav_pos,
        user_pos,
        num_samples=60
    ):
        """
        判断 UAV 到地面用户之间是否具有直视链路。

        使用线段采样法：
        沿 UAV-user 连线取若干采样点，
        若任一点落入建筑物内部，则判定为 NLoS。
        """

        uav_pos = np.asarray(
            uav_pos,
            dtype=np.float64
        )

        user_3d = np.array(
            [
                float(user_pos[0]),
                float(user_pos[1]),
                0.0
            ],
            dtype=np.float64
        )

        # 不采样两个端点，避免用户或 UAV 恰好位于边界
        alphas = np.linspace(
            0.0,
            1.0,
            num_samples + 2
        )[1:-1]

        for alpha in alphas:
            point = (
                uav_pos
                + alpha * (user_3d - uav_pos)
            )

            point_x = float(point[0])
            point_y = float(point[1])
            point_z = float(point[2])

            for building in self.buildings:
                inside_horizontal_footprint = (
                    building["x_min"]
                    <= point_x
                    <= building["x_max"]
                    and building["y_min"]
                    <= point_y
                    <= building["y_max"]
                )

                below_building_height = (
                    point_z
                    < building["height"]
                )

                if (
                    inside_horizontal_footprint
                    and below_building_height
                ):
                    return False

        return True
   
    def step(self, action):
        """
        Action definition:
        - action[0]: x-direction movement preference
        - action[1]: y-direction movement preference
        - action[2]: vertical movement preference
        - action[3]: total transmit-power control
        - action[4:4 + num_users]: per-user power-allocation preferences
        - remaining padded power slots are unused
        """

        # ============================================================
        # 0. 输入检查与时间推进
        # ============================================================
        action = np.asarray(
            action,
            dtype=np.float32
        ).reshape(-1)

        expected_action_dim = (
            4 + self.MAX_USERS_LIMIT
        )

        if action.shape[0] != expected_action_dim:
            raise ValueError(
                f"Action dimension mismatch: "
                f"expected {expected_action_dim}, "
                f"got {action.shape[0]}"
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        ).astype(np.float32)

        self.current_step += 1

        # ============================================================
        # 1. V5.6 Smoothed UAV movement control
        # ============================================================

        max_horizontal_speed = 5.0
        max_vertical_speed = 2.0

        # PPO 当前输出的原始移动动作
        raw_movement_action = np.asarray(
            action[:3],
            dtype=np.float32
        )

        # 一阶低通滤波：
        # 实际执行动作由上一时刻动作和当前 PPO 动作共同决定
        movement_beta = float(
            self.MOVEMENT_SMOOTHING_BETA
        )

        applied_movement_action = (
            (1.0 - movement_beta)
            * self.previous_movement_action
            + movement_beta
            * raw_movement_action
        ).astype(np.float32)

        # 对平滑后的极小动作设置轻微死区
        dead_zone = float(
            self.MOVEMENT_DEAD_ZONE
        )

        applied_movement_action[
            np.abs(applied_movement_action) < dead_zone
        ] = 0.0

        # 确保动作仍处于合法范围
        applied_movement_action = np.clip(
            applied_movement_action,
            -1.0,
            1.0
        ).astype(np.float32)

        # 使用实际平滑动作计算 UAV 速度
        v_x = (
            float(applied_movement_action[0])
            * max_horizontal_speed
        )

        v_y = (
            float(applied_movement_action[1])
            * max_horizontal_speed
        )

        v_z = (
            float(applied_movement_action[2])
            * max_vertical_speed
        )
        # ============================================================
        # V5.6 Adaptive altitude hold
        # ============================================================

        current_altitude = float(
            self.uav_pos[2]
        )

        vertical_action = float(
            applied_movement_action[2]
        )

        altitude_hold_active = False
        altitude_hold_error = 0.0

        # ------------------------------------------------------------
        # 1. PPO 给出明确升降指令时，解除高度保持
        # ------------------------------------------------------------

        if (
            self.altitude_hold_target is not None
            and abs(vertical_action)
            >= self.ALTITUDE_HOLD_RELEASE_THRESHOLD
        ):
            self.altitude_hold_target = None

        # ------------------------------------------------------------
        # 2. 高度达到较高区域，并且垂直动作已经很小时，
        #    将当前高度记录为保持目标
        # ------------------------------------------------------------

        if (
            self.altitude_hold_target is None
            and current_altitude
            >= self.ALTITUDE_HOLD_MIN_ALTITUDE
            and abs(vertical_action)
            <= self.ALTITUDE_HOLD_ACTION_THRESHOLD
        ):
            self.altitude_hold_target = (
                current_altitude
            )

        # ------------------------------------------------------------
        # 3. 高度保持、释放旧目标
        # ------------------------------------------------------------

        if self.altitude_hold_target is not None:
            altitude_hold_error = float(
                self.altitude_hold_target
                - current_altitude
            )

            # 如果已经离开原来的高度保持带，
            # 说明 UAV 正在选择新的高度，释放旧目标。
            if (
                abs(altitude_hold_error)
                > self.ALTITUDE_HOLD_BAND
            ):
                self.altitude_hold_target = None
                altitude_hold_error = 0.0

            else:
                small_vertical_command = (
                    abs(vertical_action)
                    <= self.ALTITUDE_HOLD_ACTION_THRESHOLD
                )

                if small_vertical_command:
                    altitude_hold_active = True

                    # 取消实际垂直速度
                    v_z = 0.0

                    # 下一步 observation 和滤波器看到的
                    # 也是实际执行后的零垂直动作
                    applied_movement_action[2] = 0.0

        previous_pos = self.uav_pos.copy()

        v_horiz = float(
            np.sqrt(v_x ** 2 + v_y ** 2)
        )

        v_3d = float(
            np.sqrt(
                v_x ** 2
                + v_y ** 2
                + v_z ** 2
            )
        )

        # 裁剪前目标位置，用于检测越界
        raw_next_x = (
            float(self.uav_pos[0])
            + v_x * self.dt
        )

        raw_next_y = (
            float(self.uav_pos[1])
            + v_y * self.dt
        )

        raw_next_z = (
            float(self.uav_pos[2])
            + v_z * self.dt
        )

        altitude_violation = bool(
            raw_next_z < self.MIN_ALTITUDE
            or raw_next_z > self.MAX_ALTITUDE
        )
        boundary_violation = bool(
            raw_next_x < 0.0
            or raw_next_x > self.MAP_LIMIT
            or raw_next_y < 0.0
            or raw_next_y > self.MAP_LIMIT
        )

        proposed_position = np.array(
            [
                np.clip(
                    raw_next_x,
                    0.0,
                    self.MAP_LIMIT
                ),
                np.clip(
                    raw_next_y,
                    0.0,
                    self.MAP_LIMIT
                ),
                np.clip(
                    raw_next_z,
                    self.MIN_ALTITUDE,
                    self.MAX_ALTITUDE
                )
            ],
            dtype=np.float32
        )

        building_collision = self.is_inside_building(
            proposed_position
        )

        # 如果下一位置进入建筑，则保持原位置
        if building_collision:
            self.uav_pos = previous_pos.copy()
        else:
            self.uav_pos = proposed_position.copy()

        actual_displacement = float(
            np.linalg.norm(
                self.uav_pos
                - previous_pos
            )
        )
        horizontal_displacement = float(
            np.linalg.norm(
                self.uav_pos[:2]
                - previous_pos[:2]
            )
        )

        vertical_displacement = float(
            abs(
                self.uav_pos[2]
                - previous_pos[2]
            )
        )
        # ============================================================
        # V5.4 User mobility update
        # ============================================================
        if self.active_mobility == "evacuation":

            self._update_evacuation_users()

        elif self.active_mobility == "straggler":

            self._update_straggler_users()
        elif self.active_mobility == "random_walk":

            self._update_random_walk_users()
        
        # ============================================================
        # 2. V4.3 可变总发射功率与多用户功率分配
        # ============================================================

        # ------------------------------------------------------------
        # 2.1 PPO 决定当前 timestep 的总发射功率
        # ------------------------------------------------------------
        # action[3] ∈ [-1, 1]
        # 线性映射到 [P_min, P_max]
        transmit_power_fraction = float(
            (action[3] + 1.0) / 2.0
        )

        transmit_power_fraction = float(
            np.clip(
                transmit_power_fraction,
                0.0,
                1.0
            )
        )

        transmit_power_budget = float(
            self.P_min
            + transmit_power_fraction
            * (
                self.P_max
                - self.P_min
            )
        )

        # ------------------------------------------------------------
        # 2.2 当前总功率中的保底部分
        # ------------------------------------------------------------
        # 当前总发射功率的 15% 均匀分给所有用户，
        # 剩余 85% 由 PPO 根据 softmax 权重分配。
        minimum_power_fraction = 0.15

        total_floor_power = float(
            minimum_power_fraction
            * transmit_power_budget
        )

        p_floor = float(
            total_floor_power
            / self.num_users
        )

        p_dynamic_pool = float(
            transmit_power_budget
            - total_floor_power
        )

        if self.power_allocation_mode == "equal":

            # Group C:
            # PPO不控制用户级功率比例，
            # 每个用户始终获得相同功率比例。
            raw_power_weights = np.full(
                self.num_users,
                1.0 / self.num_users,
                dtype=np.float32
            )

        else:

            # Group A:
            # PPO学习用户级功率分配偏好。
            power_preferences = action[
                4:4 + self.num_users
            ].copy()

            shifted_preferences = (
                power_preferences
                - np.max(power_preferences)
            )

            exp_preferences = np.exp(
                shifted_preferences
            )

            preference_sum = float(
                np.sum(exp_preferences)
            )

            if (
                not np.isfinite(preference_sum)
                or preference_sum <= 0.0
            ):
                raw_power_weights = np.full(
                    self.num_users,
                    1.0 / self.num_users,
                    dtype=np.float32
                )
            else:
                raw_power_weights = (
                    exp_preferences
                    / preference_sum
                ).astype(np.float32)

        # ============================================================
        # V5.2 Exponential smoothing
        # ============================================================

        beta = float(
            self.POWER_ALLOCATION_SMOOTHING_BETA
        )

        power_weights = (
            (1.0 - beta)
            * self.previous_power_weights
            + beta
            * raw_power_weights
        )

        # 数值保护：确保权重严格非负且总和为1
        power_weights = np.maximum(
            power_weights,
            0.0
        )

        power_weight_sum = float(
            np.sum(power_weights)
        )

        if (
            not np.isfinite(power_weight_sum)
            or power_weight_sum <= 0.0
        ):
            power_weights = np.full(
                self.num_users,
                1.0 / self.num_users,
                dtype=np.float32
            )
        else:
            power_weights = (
                power_weights
                / power_weight_sum
            ).astype(np.float32)

        # floor power 保底部分保持不变；
        # 只对 dynamic pool 的 PPO 权重进行平滑。
        p_allocated = (
            p_floor
            + power_weights
            * p_dynamic_pool
        ).astype(np.float32)

        # ------------------------------------------------------------
        # 2.4 数值保护：精确满足当前总功率预算
        # ------------------------------------------------------------
        allocated_power_sum = float(
            np.sum(p_allocated)
        )

        if allocated_power_sum <= 0.0:
            p_allocated = np.full(
                self.num_users,
                transmit_power_budget
                / self.num_users,
                dtype=np.float32
            )
        else:
            p_allocated *= (
                transmit_power_budget
                / allocated_power_sum
            )

        transmit_power = float(
            np.sum(p_allocated)
        )
        # ============================================================
        # V5.2 Power-allocation stability metric
        # ============================================================

        power_allocation_fractions = (
            p_allocated
            / max(
                transmit_power_budget,
                1e-8
            )
        ).astype(np.float32)

        # Total variation distance:
        # 0 表示功率比例完全没有变化；
        # 最大值约为1。
        power_allocation_change_penalty = float(
            0.5
            * np.sum(
                np.abs(
                    power_allocation_fractions
                    - self.previous_power_allocation_fractions
                )
            )
        )
        # ============================================================
        # 3. UAV 功耗模型
        # ============================================================
        # 第一阶段使用轻量、单调、可解释的功耗模型：
        # 悬停基础功耗 + 水平移动附加功耗
        P_hover = 0.35

        horizontal_power_coefficient = 0.015

        P_horizontal = ( horizontal_power_coefficient * v_horiz ** 2)

        vertical_climb_coefficient = 0.06
        vertical_descent_coefficient = 0.03

        P_vertical = (
            vertical_climb_coefficient
            * max(v_z, 0.0)
            +
            vertical_descent_coefficient
            * max(-v_z, 0.0)
        )

        P_aero = (
            P_hover
            + P_horizontal
            + P_vertical
        )

        # 通信电子电路基础功耗
        P_circuit = 0.05

        P_elec = (
            P_circuit
            + transmit_power
        )

        total_power = float(
            P_aero + P_elec
        )

        energy_consumed = (
            total_power * self.dt
        )

        self.battery -= energy_consumed

        # 避免观测中出现负电池值
        self.battery = max(
            0.0,
            float(self.battery)
        )

        # ============================================================
        # 4. 无线信道与用户速率
        # ============================================================
        self.current_rates = np.zeros(
            self.num_users,
            dtype=np.float64
        )

        user_distances = np.zeros(
            self.num_users,
            dtype=np.float64
        )

        user_snrs = np.zeros(
            self.num_users,
            dtype=np.float64
        )

        user_sinrs = np.zeros(
            self.num_users,
            dtype=np.float32
        )

        user_interference_powers = np.zeros(
            self.num_users,
            dtype=np.float32
        )

        user_los_flags = np.zeros(
            self.num_users,
            dtype=bool
        )

        user_path_losses_db = np.zeros(
            self.num_users,
            dtype=np.float64
        )

        for i in range(self.num_users):
            user_3d_position = np.array(
                [
                    self.user_positions[i, 0],
                    self.user_positions[i, 1],
                    0.0
                ],
                dtype=np.float64
            )

            d_3d = float(
                np.linalg.norm(
                    self.uav_pos.astype(
                        np.float64
                    )
                    - user_3d_position
                )
            )

            d_3d = max(
                d_3d,
                1.0
            )

            user_distances[i] = d_3d

                        # 判断当前 UAV-user 链路是 LoS 还是 NLoS
            is_los = self.check_line_of_sight(
                self.uav_pos,
                self.user_positions[i]
            )

            user_los_flags[i] = is_los

            free_space_path_loss_db = (
                20.0 * np.log10(d_3d)
                + 20.0 * np.log10(self.f_c)
                - 147.55
            )

            additional_loss_db = (
                self.LOS_EXTRA_LOSS_DB
                if is_los
                else self.NLOS_EXTRA_LOSS_DB
            )

            path_loss_db = (
                free_space_path_loss_db
                + additional_loss_db
            )

            user_path_losses_db[i] = path_loss_db

            channel_gain = (
                10.0
                ** (-path_loss_db / 10.0)
            )

            # ============================================================
            # Desired received signal power
            # ============================================================
            desired_signal_power = (
                float(p_allocated[i])
                * channel_gain
            )

            # ============================================================
            # Total transmit power allocated to the other users
            # ============================================================
            other_user_transmit_power = float(
                transmit_power
                - p_allocated[i]
            )

            # ============================================================
            # Residual multi-user interference power
            # ============================================================
            if self.use_interference:
                interference_power = float(
                    self.interference_coupling
                    * channel_gain
                    * max(
                        other_user_transmit_power,
                        0.0
                    )
                )
            else:
                interference_power = 0.0

            # ============================================================
            # SNR: interference-free diagnostic metric
            # ============================================================
            snr = float(
                desired_signal_power
                / self.sigma2
            )

            # ============================================================
            # SINR: effective link quality with interference
            # ============================================================
            sinr = float(
                desired_signal_power
                / (
                    interference_power
                    + self.sigma2
                )
            )

            # 数值保护，防止出现0或负数
            snr = max(
                snr,
                1e-12
            )

            sinr = max(
                sinr,
                1e-12
            )

            user_snrs[i] = snr


            user_sinrs[i] = sinr

            user_interference_powers[i] = (
                interference_power
            )
            # 使用 SINR 计算用户速率
            self.current_rates[i] = (
                self.B
                * np.log2(
                    1.0 + sinr
                )
            )

        # ============================================================
        # Per-step communication summary
        # ============================================================

        self.user_rates_history = (
            self.current_rates.copy()
        )

        los_user_count = int(
            np.sum(
                user_los_flags
            )
        )

        nlos_user_count = int(
            self.num_users
            - los_user_count
        )

        los_ratio = float(
            los_user_count
            / self.num_users
        )

        self.current_los_flags = (
            user_los_flags.astype(
                np.float32
            ).copy()
        )
        # ============================================================
        # 5. 通信与公平性指标
        # ============================================================
        total_throughput = float(
            np.sum(self.current_rates)
        )

        avg_rate = float(
            np.mean(self.current_rates)
        )

        min_rate = float(
            np.min(self.current_rates)
        )

        max_rate = float(
            np.max(self.current_rates)
        )

        spectral_efficiency_sum = float(
            total_throughput / self.B
        )

        rates_square_sum = float(
            np.sum(
                self.current_rates ** 2
            )
        )

        if rates_square_sum <= 0.0:
            jain_index = 0.0
        else:
            jain_index = float(
                total_throughput ** 2
                / (
                    self.num_users
                    * rates_square_sum
                    + 1e-12
                )
            )

        # ============================================================
        # 6. Energy Efficiency
        # ============================================================
        energy_efficiency = float(
            total_throughput
            / (total_power + 1e-9)
        )

        # 将数亿到十亿量级 EE 映射到较稳定的 reward 范围
        EE_REF = 1.0e9

        ee_score = float(
            np.log1p(
                energy_efficiency / EE_REF
            )
        )

               # ============================================================
        # 7. QoS constraints
        # ============================================================

        # 保留你当前正在使用的 QoS 门槛，不在本次实验中改动
        R_MIN = 15.0e6

        rate_deficit = np.maximum(
            0.0,
            R_MIN - self.current_rates
        )

        normalized_rate_deficit = (
            rate_deficit / R_MIN
        )

        # 平均 QoS 缺口：
        # 反映整个用户群总体有多严重地不满足 QoS
        mean_qos_penalty = float(
            np.mean(
                normalized_rate_deficit
            )
        )

        # 最差用户 QoS 缺口：
        # 防止平均值掩盖某一个严重掉线用户
        worst_user_qos_penalty = float(
            np.max(
                normalized_rate_deficit
            )
        )

        qos_violation_count = int(
            np.sum(
                self.current_rates < R_MIN
            )
        )

        qos_violation_ratio = float(
            qos_violation_count
            / self.num_users
        )

        # 当前最差用户的索引，便于分析
        worst_user_index = int(
            np.argmin(
                self.current_rates
            )
        )
        # ============================================================
        # 8. V5.6 动作平滑、移动与方向反转惩罚
        # ============================================================

        # 使用环境真正执行的平滑动作计算惩罚，
        # 而不是使用 PPO 尚未完全执行的原始动作
        current_movement_action = (
            applied_movement_action.copy()
        )

        previous_movement_action = np.asarray(
            self.previous_movement_action,
            dtype=np.float32
        )

        action_delta = (
            current_movement_action
            - previous_movement_action
        )

        smooth_penalty = float(
            np.mean(
                action_delta ** 2
            )
        )

        # ============================================================
        # V5.6 Direction-reversal penalty
        # ============================================================

        current_horizontal_action = np.asarray(
            current_movement_action[:2],
            dtype=np.float32
        )

        previous_horizontal_action = np.asarray(
            previous_movement_action[:2],
            dtype=np.float32
        )

        current_horizontal_norm = float(
            np.linalg.norm(
                current_horizontal_action
            )
        )

        previous_horizontal_norm = float(
            np.linalg.norm(
                previous_horizontal_action
            )
        )

        if (
            current_horizontal_norm > 1e-6
            and previous_horizontal_norm > 1e-6
        ):
            movement_direction_cosine = float(
                np.dot(
                    current_horizontal_action,
                    previous_horizontal_action
                )
                / (
                    current_horizontal_norm
                    * previous_horizontal_norm
                    + 1e-8
                )
            )

            # 同方向时 cosine > 0，惩罚为0；
            # 反方向时 cosine < 0，产生正惩罚。
            direction_reversal_penalty = float(
                max(
                    0.0,
                    -movement_direction_cosine
                )
                * min(
                    current_horizontal_norm,
                    previous_horizontal_norm
                )
            )
        else:
            movement_direction_cosine = 1.0
            direction_reversal_penalty = 0.0

        # 归一化移动强度：
        # 单轴最大速度时约为1，
        # 对角最大速度时可到2
        horizontal_movement_penalty = float(
            (
                v_horiz
                / max_horizontal_speed
            ) ** 2
        )

        vertical_movement_penalty = float(
            (
                abs(v_z)
                / max_vertical_speed
            ) ** 2
        )

        movement_penalty = (
            horizontal_movement_penalty
            + 0.5 * vertical_movement_penalty
        )

        # ============================================================
        # 9. 软边界和硬越界惩罚
        # ============================================================
        boundary_margin = 30.0

        distance_to_nearest_boundary = float(
            min(
                self.uav_pos[0],
                self.MAP_LIMIT
                - self.uav_pos[0],
                self.uav_pos[1],
                self.MAP_LIMIT
                - self.uav_pos[1]
            )
        )

        soft_boundary_penalty = float(
            max(
                0.0,
                (
                    boundary_margin
                    - distance_to_nearest_boundary
                )
                / boundary_margin
            )
        )

        hard_boundary_penalty = float(
            boundary_violation
        )

        # ============================================================
        # 10. V5.1 动态用户群跟踪指标
        # ============================================================
        centroid = np.mean(
            self.user_positions,
            axis=0
        )

        dist_to_centroid = float(
            np.linalg.norm(
                self.uav_pos[:2]
                - centroid
            )
        )

        # 正值：UAV 比上一时刻更接近用户质心
        # 负值：UAV 正在被用户群甩开
        tracking_progress = float(
            self.previous_tracking_distance
            - dist_to_centroid
        )

        # 距离归一化到约 [0, 1]
        normalized_tracking_distance = float(
            np.clip(
                dist_to_centroid / self.MAP_LIMIT,
                0.0,
                1.0
            )
        )

        # 以每步 UAV 最大水平位移作为尺度
        normalized_tracking_progress = float(
            np.clip(
                tracking_progress
                / (
                    max_horizontal_speed
                    * self.dt
                    + 1e-6
                ),
                -1.0,
                1.0
            )
        )

        # ============================================================
        # 11. V5.1 正式 Reward
        # ============================================================

        EE_WEIGHT = 2.0

        MEAN_QOS_WEIGHT = 2.0
        WORST_USER_QOS_WEIGHT = 3.5

        LOS_COVERAGE_WEIGHT = 0.20

        TRACKING_DISTANCE_WEIGHT = 1.20
        TRACKING_PROGRESS_WEIGHT = 1.00

        SMOOTH_WEIGHT = 0.10
        MOVEMENT_WEIGHT = 0.06
        DIRECTION_REVERSAL_WEIGHT = float(
            self.DIRECTION_REVERSAL_WEIGHT
        )

        SOFT_BOUNDARY_WEIGHT = 0.50
        HARD_BOUNDARY_WEIGHT = 2.0
        BUILDING_COLLISION_WEIGHT = 3.0
        ALTITUDE_VIOLATION_WEIGHT = 2.0

        POWER_ALLOCATION_CHANGE_WEIGHT = float(
            self.POWER_ALLOCATION_CHANGE_WEIGHT
        )
        reward_ee_component = (
            EE_WEIGHT
            * ee_score
        )

        reward_mean_qos_component = (
            -MEAN_QOS_WEIGHT
            * mean_qos_penalty
        )

        reward_worst_user_qos_component = (
            -WORST_USER_QOS_WEIGHT
            * worst_user_qos_penalty
        )

        # 必须保留这一段
        reward_los_coverage_component = (
            LOS_COVERAGE_WEIGHT
            * los_ratio
        )

        reward_tracking_distance_component = (
            -TRACKING_DISTANCE_WEIGHT
            * normalized_tracking_distance
        )

        reward_tracking_progress_component = (
            TRACKING_PROGRESS_WEIGHT
            * normalized_tracking_progress
        )

        reward_smooth_component = (
            -SMOOTH_WEIGHT
            * smooth_penalty
        )

        reward_movement_component = (
            -MOVEMENT_WEIGHT
            * movement_penalty
        )

        reward_direction_reversal_component = (
            -DIRECTION_REVERSAL_WEIGHT
            * direction_reversal_penalty
        )

        reward_soft_boundary_component = (
            -SOFT_BOUNDARY_WEIGHT
            * soft_boundary_penalty
        )

        reward_hard_boundary_component = (
            -HARD_BOUNDARY_WEIGHT
            * hard_boundary_penalty
        )

        reward_building_collision_component = (
            -BUILDING_COLLISION_WEIGHT
            * float(building_collision)
        )

        reward_altitude_violation_component = (
            -ALTITUDE_VIOLATION_WEIGHT
            * float(altitude_violation)
        )
        reward_power_allocation_change_component = (
           - POWER_ALLOCATION_CHANGE_WEIGHT
            * power_allocation_change_penalty
        )
        reward_min_rate_component = (
            1.5
            * np.clip(
                min_rate / R_MIN,
                0.0,
                2.0
            )
        )

        self.reward = float(
            reward_ee_component
            + reward_mean_qos_component
            + reward_worst_user_qos_component
            + reward_los_coverage_component
            + reward_tracking_distance_component
            + reward_tracking_progress_component
            + reward_smooth_component
            + reward_movement_component
            + reward_direction_reversal_component
            + reward_soft_boundary_component
            + reward_hard_boundary_component
            + reward_building_collision_component
            + reward_altitude_violation_component
            + reward_min_rate_component
            + reward_power_allocation_change_component

        )

        # 数值安全检查
        if not np.isfinite(self.reward):
            raise FloatingPointError(
                "Non-finite reward detected. "
                f"EE={energy_efficiency}, "
                f"power={total_power}, "
                f"throughput={total_throughput}"
            )
        user_blocked_flags = np.array(
            [
                self.is_user_position_blocked(
                    self.user_positions[user_idx],
                    clearance=0.0
                )
                for user_idx in range(
                    self.num_users
                )
            ],
            dtype=bool
        )

        user_stationary_flags = (
            np.linalg.norm(
                self.user_velocities,
                axis=1
            )
            < 1e-6
        )
        # ============================================================
        # 12. 终止条件
        # ============================================================
        terminated = bool(
            self.battery <= 0.0
        )

        truncated = bool(
            self.current_step
            >= self.MAX_STEPS
        )

        # ============================================================
        # 13. 向评估和训练监控返回完整指标
        # ============================================================
        info_dict = {
            # 基础时间与状态
            "step": int(
                self.current_step
            ),
            "battery_level": float(
                self.battery
            ),
            "energy_consumed": float(
                energy_consumed
            ),
            "configured_mobility": self.mobility,
            "active_mobility": self.active_mobility,
            "building_layout":
                self.building_layout,

            "number_of_buildings":
                int(len(self.buildings)),
            # 位置与移动
            "uav_x": float(
                self.uav_pos[0]
            ),
            "uav_y": float(
                self.uav_pos[1]
            ),
            "uav_z": float(
                self.uav_pos[2]
            ),
            "horizontal_speed": float(
                v_horiz
            ),
            "actual_displacement": float(
                actual_displacement
            ),
            "dist": float(
                dist_to_centroid
            ),
            "tracking_distance": float(
                dist_to_centroid
            ),
            # V5.1：UAV 到动态用户群质心的跟踪距离
            # 保留 dist 兼容旧评估脚本，同时新增语义更清楚的键
            "tracking_progress": float(
                tracking_progress
            ),

            "normalized_tracking_distance": float(
                normalized_tracking_distance
            ),

            "normalized_tracking_progress": float(
                normalized_tracking_progress
            ),

            "user_centroid": (
                centroid.copy()
            ),
            "vertical_speed": float(
                v_z
            ),

            "three_dimensional_speed": float(
                v_3d
            ),

            "horizontal_displacement": float(
                horizontal_displacement
            ),

            "vertical_displacement": float(
                vertical_displacement
            ),

            "altitude": float(
                self.uav_pos[2]
            ),

            "altitude_violation": bool(
                altitude_violation
            ),
            "altitude_hold_active": bool(
                altitude_hold_active
            ),

            "altitude_hold_target": (
                float(self.altitude_hold_target)
                if self.altitude_hold_target
                is not None
                else np.nan
            ),

            "altitude_hold_error": float(
                altitude_hold_error
            ),

            "vertical_action_raw": float(
                raw_movement_action[2]
            ),

            "vertical_action_applied": float(
                applied_movement_action[2]
            ),
            # 功率
            "aerodynamic_power": float(
                P_aero
            ),
            "electronic_power": float(
                P_elec
            ),
            "transmit_power": float(
                transmit_power
            ),
            "transmit_power_budget": float(
                transmit_power_budget
            ),

            "transmit_power_fraction": float(
                transmit_power_fraction
            ),

            "minimum_transmit_power": float(
                self.P_min
            ),

            "maximum_transmit_power": float(
                self.P_max
            ),
            "total_power": float(
                total_power
            ),
            "power_allocation": (
                p_allocated.copy()
            ),

            "raw_power_weights": (
                raw_power_weights.copy()
            ),

            "smoothed_power_weights": (
                power_weights.copy()
            ),

            "power_allocation_fractions": (
                power_allocation_fractions.copy()
            ),

            "power_allocation_change_penalty": float(
                power_allocation_change_penalty
            ),
            "horizontal_propulsion_power": float(
                P_horizontal
            ),

            "vertical_propulsion_power": float(
                P_vertical
            ),
            # 通信
            "total_throughput_mbps": float(
                total_throughput / 1e6
            ),
            "avg_rate_mbps": float(
                avg_rate / 1e6
            ),
            "min_rate_mbps": float(
                min_rate / 1e6
            ),
            "max_rate_mbps": float(
                max_rate / 1e6
            ),
            "spectral_efficiency_sum": float(
                spectral_efficiency_sum
            ),
            "jain_index": float(
                jain_index
            ),
            "energy_efficiency": float(
                energy_efficiency
            ),
            
            "mean_qos_penalty": float(
                mean_qos_penalty
            ),
            "user_rates_mbps": (
                self.current_rates.copy()
                / 1e6
            ),

            "user_distances": (
                user_distances.copy()
            ),

            "qos_threshold_mbps": float(
                R_MIN / 1e6
            ),
            "worst_user_qos_penalty": float(
                worst_user_qos_penalty
            ),
            "user_snrs": user_snrs.copy(),
            "user_sinrs": user_sinrs.copy(),
            "user_interference_powers": user_interference_powers.copy(),
            "use_interference": self.use_interference,
            "interference_coupling": self.interference_coupling,
            "power_allocation_mode":
                self.power_allocation_mode,
            # 为兼容旧监控脚本，保留 qos_penalty 键
            "qos_penalty": float(
                mean_qos_penalty
            ),

            "qos_violation_count": int(
                qos_violation_count
            ),

            "qos_violation_ratio": float(
                qos_violation_ratio
            ),

            "worst_user_index": int(
                worst_user_index
            ),

            "worst_user_rate_mbps": float(
                min_rate / 1e6
            ),

            "los_ratio": float(
                los_ratio
            ),

            "los_user_count": int(
                los_user_count
            ),

            "nlos_user_count": int(
                nlos_user_count
            ),
            # 边界
            "boundary_violation": bool(
                boundary_violation
            ),
            "building_collision": bool(
                building_collision
            ),

            

            "los_flags": (
                user_los_flags.copy()
            ),

            "user_path_losses_db": (
                user_path_losses_db.copy()
            ),
            "distance_to_nearest_boundary": float(
                distance_to_nearest_boundary
            ),
            "soft_boundary_penalty": float(
                soft_boundary_penalty
            ),

            # Reward 分项
            "reward_total": float(
                self.reward
            ),
            "ee_score": float(
                ee_score
            ),


            "smooth_penalty": float(
                smooth_penalty
            ),
            "movement_penalty": float(
                movement_penalty
            ),

            "raw_movement_action": (
                raw_movement_action.copy()
            ),

            "applied_movement_action": (
                applied_movement_action.copy()
            ),

            "movement_direction_cosine": float(
                movement_direction_cosine
            ),

            "direction_reversal_penalty": float(
                direction_reversal_penalty
            ),

            "reward_direction_reversal_component": float(
                reward_direction_reversal_component
            ),
            "reward_min_rate_component": float(
                reward_min_rate_component
            ),
            "reward_ee_component": float(
                reward_ee_component
            ),
            "reward_mean_qos_component": float(
                reward_mean_qos_component
            ),

            "reward_worst_user_qos_component": float(
                reward_worst_user_qos_component
            ),

            "reward_los_coverage_component": float(
                reward_los_coverage_component
            ),
            "reward_tracking_distance_component": float(
                reward_tracking_distance_component
            ),

            "reward_tracking_progress_component": float(
                reward_tracking_progress_component
            ),
            "reward_smooth_component": float(
                reward_smooth_component
            ),
            "reward_movement_component": float(
                reward_movement_component
            ),

            "reward_power_allocation_change_component": float(
                reward_power_allocation_change_component
            ),
            "reward_soft_boundary_component": float(
                reward_soft_boundary_component
            ),
            "reward_hard_boundary_component": float(
                reward_hard_boundary_component
            ),

            "reward_building_collision_component": float(
                reward_building_collision_component
            ),
            "user_positions":
                self.user_positions.copy(),

            "user_velocities":
                self.user_velocities.copy(),

            "user_blocked_flags": (
                user_blocked_flags.copy()
            ),

            "user_stationary_flags": (
                user_stationary_flags.copy()
            ),

            "user_blocked_count": int(
                np.sum(user_blocked_flags)
            ),

            "user_stationary_count": int(
                np.sum(user_stationary_flags)
            ),
            "normalized_user_rates_obs": (
                np.clip(
                    self.current_rates
                    / 100.0e6,
                    0.0,
                    2.0
                ).astype(np.float32)
            ),

            "observation_los_flags": (
                self.current_los_flags.copy()
            ),
            "evacuation_target": (
                None
                if self.evacuation_target is None
                else self.evacuation_target.copy()
            ),

            "reward_altitude_violation_component": float(
                reward_altitude_violation_component
            )
        }

        # ============================================================
        # 14. 更新历史状态
        # ============================================================
        self.last_action = (
            action.copy()
        )


        # 保存当前实际执行的移动动作，
        # 供下一 timestep 的低通滤波和反转判断使用
        self.previous_movement_action = (
            applied_movement_action.copy()
        )

        self.last_pos = (
            self.uav_pos.copy()
        )

        self.previous_tracking_distance = float(
            dist_to_centroid
        )
        # V5.2: 保存本 timestep 实际采用的功率分配
        self.previous_power_weights = (
            power_weights.copy()
        )

        self.previous_power_allocation_fractions = (
            power_allocation_fractions.copy()
        )

        # observation 使用 float32
        observation = self._get_obs()

        return (
            observation,
            float(self.reward),
            terminated,
            truncated,
            info_dict
        )