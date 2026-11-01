"""ブロック受け渡し環境の設定"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .robot_cfg import HAND_BODY_NAME, JOINT_NAMES, SO101_URDF_PATH, make_so101_cfg
from .t_task_cfg import (
    BASE_DISTANCE,
    BLOCK_MASS,
    BLOCK_SIZE,
    GRIP_OFFSET,
    GRIPPER_OPEN_WIDE,
    GRIPPER_TOUCH,
    REST_POSE,
)

# 渡す側の提示姿勢を定義
GIVER_PRESENT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.57,
    "elbow_flex": 1.24,
    "wrist_flex": -0.68,
    "wrist_roll": 1.62,
    "gripper": 0.0,
}

# 受け取る側の初期姿勢を定義
RECEIVER_INIT_POS = {**REST_POSE, "gripper": GRIPPER_OPEN_WIDE}


@configclass
class HandoverEnvCfg(DirectRLEnvCfg):
    # 制御周期を設定
    decimation = 4  # 30 Hz
    episode_length_s = 8.0

    # 空間の次元を設定
    action_space = 5
    observation_space = 35
    state_space = 0

    # シミュレーションと接触バッファを設定
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**22,
            gpu_max_rigid_contact_count=2**23,
            gpu_found_lost_pairs_capacity=2**22,
        ),
    )

    # シーンを設定
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=True
    )

    # ロボットを配置
    base_distance = BASE_DISTANCE
    giver_cfg: ArticulationCfg = make_so101_cfg(
        prim_path="/World/envs/env_.*/Giver",
        pos=(0.0, BASE_DISTANCE / 2, 0.0),
        yaw_deg=-90.0,
        init_joint_pos=GIVER_PRESENT_POS,
        actuator="real",
    )
    receiver_cfg: ArticulationCfg = make_so101_cfg(
        prim_path="/World/envs/env_.*/Receiver",
        pos=(0.0, -BASE_DISTANCE / 2, 0.0),
        yaw_deg=90.0,
        init_joint_pos=RECEIVER_INIT_POS,
        actuator="real",
    )

    # キネマティックなブロックを設定
    block_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )

    # 関節とリンクの名前を設定
    joint_names = JOINT_NAMES
    hand_body_name = HAND_BODY_NAME
    urdf_path = SO101_URDF_PATH

    # TCP を閉じた爪の中心に設定 [m]
    tcp_offset = (0.0075, 0.0, -0.098)

    # 渡す側の把持の向きを設定
    block_in_tcp_quat = (0.7071068, 0.0, 0.0, 0.7071068)
    grip_offset = GRIP_OFFSET  # [m]

    # 渡す側の姿勢のばらつきを設定 [rad]
    giver_present_pos = GIVER_PRESENT_POS
    giver_present_noise = (0.15, 0.1, 0.1, 0.1, 0.2, 0.0)
    giver_gripper_closed = 0.2
    giver_gripper_open = 0.6

    # 受け取る側の初期姿勢のばらつきを設定 [rad]
    receiver_init_pos = RECEIVER_INIT_POS
    receiver_init_noise = 0.05

    # 行動のスケールを設定 [rad]
    action_scale = 0.05

    # 参照軌道を設定（受け取る手前 → 受け取る → 引き寄せる）
    use_reference = True
    ref_pre_time = 1.2           # [s]
    ref_recv_time = 0.8          # [s]
    ref_carry_time = 1.0         # [s]
    pre_distance = 0.06          # [m]
    residual_limit = 0.15        # [rad]
    ik_iters = 2
    # 受け取り方を設定（"side": 上端を横から、"top": 上端を真上から）
    grasp_mode = "side"
    block_size = BLOCK_SIZE
    top_grasp_depth = 0.015      # [m] 上面からの深さ
    top_pre_up = 0.02            # [m]
    top_rot_weight = 0.05
    # 姿勢の初期値を設定 [rad]
    pre_pose = (0.02, -1.631, 1.301, 0.33, -1.522)
    recv_pose = (0.016, -0.783, 1.044, -0.261, -1.522)
    carry_pose = (-0.038, -1.621, 1.279, 0.342, -1.522)

    # 把持と受け渡しの判定を設定
    grasp_window = (0.008, 0.010, 0.010)  # [m]
    grasp_align_cos = 0.94    # 約 20 度
    squeeze_margin = 0.05
    grasp_center_x = 0.0  # [m]
    gripper_closed_max = 0.32  # [rad]
    gripper_stall_vel = 0.3        # [rad/s]
    gripper_release_margin = 0.05  # [rad]
    release_steps = 3
    gripper_open_angle = GRIPPER_OPEN_WIDE   # 開く目標
    gripper_close_target = GRIPPER_TOUCH + 0.02  # 閉じる目標
    success_hold_steps = 30

    # 引き寄せ位置を設定 [m]
    carry_target = (0.0, -0.06, 0.17)
    carry_tol = 0.03          # [m]
    carry_std = 0.1
    min_hand_clearance = 0.025

    # 報酬の重みを設定
    reach_std = 0.1
    rew_reach = 1.0
    rew_align = 0.5
    rew_in_jaws = 2.0
    rew_grasp = 1.0
    rew_release = 5.0
    rew_hold = 1.0
    rew_carry = 3.0
    rew_success = 20.0
    rew_drop = -10.0
    rew_clearance = -1.0
    rew_action_rate = -0.01
    rew_joint_vel = -0.0005

    # 集計するエピソード数を設定
    outcome_window = 1024

    def __post_init__(self):
        # ロボットの位置を設定
        half = self.base_distance * 0.5
        self.giver_cfg.init_state.pos = (0.0, half, 0.0)
        self.receiver_cfg.init_state.pos = (0.0, -half, 0.0)


# 青を低い位置で差し出す姿勢を定義（立てた T の縦棒用）
GIVER_LOW_POS = {
    "shoulder_pan": 0.024,
    "shoulder_lift": 0.851,
    "elbow_flex": 0.621,
    "wrist_flex": -1.472,
    "wrist_roll": 1.619,
    "gripper": 0.0,
}


@configclass
class HandoverTopEnvCfg(HandoverEnvCfg):
    # 低い位置の上端を真上からつまみ、立てる位置の上まで運ぶ
    grasp_mode = "top"
    giver_present_pos = GIVER_LOW_POS
    giver_present_noise = (0.08, 0.05, 0.05, 0.05, 0.2, 0.0)
    pre_distance = 0.04          # [m]
    pre_pose = (-0.001, -0.25, 0.166, 1.638, 0.048)
    recv_pose = (-0.001, 0.093, -0.076, 1.555, 0.048)
    carry_pose = (0.811, -0.243, 0.234, 1.58, 0.86)
    carry_target = (0.12, -0.14, 0.065)

    def __post_init__(self):
        super().__post_init__()
        self.giver_cfg.init_state.joint_pos = dict(GIVER_LOW_POS)
