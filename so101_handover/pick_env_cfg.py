"""供給側の把持・差し出し環境の設定。"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .robot_cfg import HAND_BODY_NAME, JOINT_NAMES, SO101_URDF_PATH, make_so101_cfg
from .t_task_cfg import (
    BASE_DISTANCE,
    BLOCK_HALF_HEIGHT,
    BLOCK_MASS,
    BLOCK_SIZE,
    BLUE_START,
    FEEDER_REST_POSE,
    GRIP_OFFSET,
    GRIPPER_OPEN,
    GRIPPER_TOUCH,
    HANDOVER_CENTER,
    RANDOM_XY,
    RANDOM_YAW_DEG,
    RED_START,
    REST_POSE,
    STAND_HANDOVER_CENTER,
)

# 供給側の初期姿勢を定義
FEEDER_INIT_POS = {**FEEDER_REST_POSE, "gripper": GRIPPER_OPEN}

# 差し出す姿勢を定義 [rad]
PRESENT_POSE = (0.0, -0.57, 1.24, -0.68, 1.62)
LOW_PRESENT_POSE = (0.024, 0.851, 0.621, -1.472, 1.619)


@configclass
class PickEnvCfg(DirectRLEnvCfg):
    # 制御周期を設定
    decimation = 4
    episode_length_s = 12.0

    # 空間の次元を設定
    action_space = 5
    observation_space = 49
    state_space = 0

    # シミュレーションを設定
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
    feeder_cfg: ArticulationCfg = make_so101_cfg(
        prim_path="/World/envs/env_.*/Feeder",
        pos=(0.0, BASE_DISTANCE / 2, 0.0),
        yaw_deg=-90.0,
        init_joint_pos=FEEDER_INIT_POS,
        actuator="real",
    )
    builder_cfg: ArticulationCfg = make_so101_cfg(
        prim_path="/World/envs/env_.*/Builder",
        pos=(0.0, -BASE_DISTANCE / 2, 0.0),
        yaw_deg=90.0,
        init_joint_pos=REST_POSE,
        actuator="real",
    )

    # ブロックを設定（キネマティック、衝突なし）
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=RED_START),
    )

    # 関節とリンクの名前を設定
    joint_names = JOINT_NAMES
    hand_body_name = HAND_BODY_NAME
    jaw_body_name = "moving_jaw_so101_v1_link"
    gripper_joint_name = "gripper"
    urdf_path = SO101_URDF_PATH

    # TCP を閉じた爪の中心に設定（gripper_link 座標系）
    tcp_offset = (0.0075, 0.0, -0.098)

    # ブロックの置き場所を設定
    block_starts = (BLUE_START, RED_START)
    block_start_yaw_deg = 90.0
    block_random_xy = RANDOM_XY
    block_random_yaw_deg = RANDOM_YAW_DEG
    block_half_height = BLOCK_HALF_HEIGHT
    block_size = BLOCK_SIZE

    # 掴む位置のオフセットを設定
    grip_offset = GRIP_OFFSET

    # 差し出す姿勢を設定
    present_center = HANDOVER_CENTER
    present_approach = (0.0, -1.0, 0.0)

    # 供給側の初期姿勢のばらつきを設定 [rad]
    feeder_init_pos = FEEDER_INIT_POS
    feeder_init_noise = 0.05

    # 真上から始める割合と姿勢を設定 [rad]
    hover_start_prob = 0.0
    hover_poses = (
        (-0.447, -0.309, 0.409, 1.471, 1.172),   # 青の置き場所 (+0.08, 0.10)
        (0.573, -0.208, 0.309, 1.470, 2.193),    # 赤の置き場所 (-0.08, 0.10)
    )

    # 把持状態から始める割合と差し出す姿勢を設定 [rad]
    attached_start_prob = 0.0
    present_pose = PRESENT_POSE

    # 参照軌道を設定
    use_reference = True
    adapt_reference = True
    hover_height = 0.05         # [m]
    ref_hover_time = 1.5        # [s]
    ref_descend_time = 1.0      # [s]
    ref_present_time = 2.0      # [s]
    present_via_hover = False
    ref_lift_time = 0.8         # [s]
    residual_mode = "direct"    # "direct" または "integrate"
    residual_limit = 0.15       # [rad]
    residual_limit_after = 0.15  # [rad]
    reset_residual_on_grasp = True
    # 掴む姿勢の初期値を設定 [rad]
    grasp_poses = (
        (-0.447, -0.197, 0.682, 1.085, 1.172),   # 青の置き場所 (+0.08, 0.10)
        (0.573, -0.099, 0.576, 1.094, 2.193),    # 赤の置き場所 (-0.08, 0.10)
    )

    # 作業範囲を設定 [m]
    area_min = (-0.18, -0.06, 0.0)
    area_max = (0.18, 0.22, 0.25)

    # 行動とグリッパを設定
    action_scale = 0.05
    gripper_open_angle = GRIPPER_OPEN
    gripper_close_target = GRIPPER_TOUCH + 0.02
    gripper_hold_angle = GRIPPER_TOUCH

    # 把持判定を設定
    grasp_window = (0.008, 0.010, 0.010)    # [m]
    grasp_align_cos = 0.9                   # 約 25 度
    grasp_down_cos = 0.8                    # 約 37 度
    gripper_closed_max = 0.30                # [rad]
    gripper_stall_vel = 0.3
    gripper_release_margin = 0.05
    squeeze_margin = 0.05
    # 固定位置のオフセットを設定 [m]
    grasp_center_x = 0.0
    grasp_steps = 2
    # 掴んだときの姿勢のまま固定する（False: 爪の間の決まった姿勢へ寄せる）
    keep_grasp_pose = False

    # 爪の食い込み判定を設定
    penetration_margin = 0.003              # [m]
    penetration_voxel = 0.005               # [m]
    grasp_reject_penetration = True

    # 差し出し判定を設定
    present_tol = 0.02                      # [m]
    present_vertical_cos = 0.95             # 約 18 度
    present_approach_cos = 0.9              # 約 26 度
    success_hold_steps = 30

    # 報酬の重みを設定
    reach_std = 0.1
    orient_std = 0.2                        # [m]
    present_std = 0.05                      # [m]
    present_std_coarse = 0.3                # [m]
    lift_height = 0.1                       # [m]
    rew_reach = 1.0
    rew_align = 1.0
    rew_in_jaws = 2.0
    rew_grasp = 5.0
    rew_hold = 1.0
    rew_lift = 2.0
    rew_present = 2.0
    rew_present_pose = 3.0
    rew_success = 20.0
    rew_drop = -10.0
    rew_out_of_area = -10.0
    rew_penetration = -1.0
    rew_residual = -0.1
    rew_action_rate = -0.01
    rew_joint_vel = -0.0005

    # 集計するエピソード数を設定
    outcome_window = 1024


@configclass
class PickLowEnvCfg(PickEnvCfg):
    # 青を低い位置に差し出す（立てた T の縦棒用）
    block_starts = (BLUE_START,)
    present_center = STAND_HANDOVER_CENTER
    present_pose = LOW_PRESENT_POSE
    present_via_hover = True
    ref_lift_time = 0.8         # [s]
    ref_present_time = 1.6      # [s]
    area_min = (-0.18, -0.10, 0.0)
