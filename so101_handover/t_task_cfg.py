"""T 字組み立てタスクのシーン設定。座標は環境原点基準 [m]。

    供給側土台 (0, +0.30)
    赤 (-0.08, +0.10)  青 (+0.08, +0.10)
    受け渡し 赤 (0, 0, 0.165) / 青 (0, -0.05, 0.055)
    平置き T (-0.08, -0.10)  立てた T (+0.12, -0.14)
    組立側土台 (0, -0.30)  D435i (-0.40, 0)
"""

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from .robot_cfg import make_so101_cfg

# ロボットの配置を設定
BASE_DISTANCE = 0.60
FEEDER_BASE = (0.0, BASE_DISTANCE / 2, 0.0)
BUILDER_BASE = (0.0, -BASE_DISTANCE / 2, 0.0)

# 休息姿勢を設定
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.70,
    "elbow_flex": 1.60,
    "wrist_flex": 0.90,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# 供給側の休息姿勢を設定
FEEDER_REST_POSE = {**REST_POSE, "shoulder_pan": -1.3}

# 可動範囲確認用の初期姿勢を設定
READY_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.6,
    "wrist_flex": 1.5,
    "wrist_roll": 0.0,
    "gripper": 0.3,
}

# ブロックの寸法と質量を設定
BLOCK_SIZE = (0.06, 0.03, 0.03)
BLOCK_MASS = 0.03
BLOCK_HALF_HEIGHT = BLOCK_SIZE[2] / 2

# ブロックの初期位置を設定
BLUE_START = (0.08, 0.10, BLOCK_HALF_HEIGHT)
RED_START = (-0.08, 0.10, BLOCK_HALF_HEIGHT)
BLOCK_START_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)

# 受け渡し位置を設定
HANDOVER_CENTER = (0.0, 0.0, 0.165)

# 縦棒（青）の配置を設定
STEM_CENTER = (-0.08, -0.10, BLOCK_HALF_HEIGHT)
STEM_DIR = (0.0, 1.0, 0.0)
# 横棒（赤）の配置を設定
BAR_CENTER = (
    STEM_CENTER[0],
    STEM_CENTER[1] + (BLOCK_SIZE[0] - BLOCK_SIZE[1]) / 2,
    BLOCK_SIZE[2] + BLOCK_HALF_HEIGHT,
)
BAR_DIR = (1.0, 0.0, 0.0)

# 把持の設定
GRIP_OFFSET = 0.019       # 中心から把持点までの距離 [m]
GRASP_LATERAL = 0.012     # 可動爪側へのずれ [m]
GRASP_DZ = -0.008         # 真上把持時の TCP 高さ [m]
HOVER_HEIGHT = 0.06       # 把持・配置前の待機高さ [m]
APPROACH_DIST = 0.03      # 水平把持前の待機距離 [m]
REORIENT_RADIUS = 0.22    # 手首を倒し終える半径 [m]
REORIENT_HEIGHT = 0.08    # 手首を倒し終える高さ [m]
RELEASE_DZ = 0.004        # 配置時の浮かせ量 [m]
GRIPPER_OPEN = 0.4        # 隙間 約 46 mm
GRIPPER_OPEN_WIDE = 0.6   # 隙間 約 62 mm
GRIPPER_TOUCH = 0.20      # 隙間 約 31 mm
FIXED_JAW_X = -0.020      # 固定爪の内面位置 [m]
ATTACH_PENETRATION = 0.002  # 許容めり込み量 [m]
HORIZONTAL_GRASP_LATERAL = 0.012  # 水平把持時の可動爪側へのずれ [m]
ATTACH_DEPTH_MIN = -0.040 # 固定判定の奥行き下限 [m]
ATTACH_DEPTH_MAX = 0.010  # 固定判定の奥行き上限 [m]
ATTACH_WIDTH = 0.010      # 固定判定の幅方向の許容量 [m]
HORIZONTAL_GRASP_DEPTH = 0.008  # 水平把持時の差し込み量 [m]
GRASP_CENTER_X = -0.0045  # 閉じた後のブロック中心位置 [m]
ATTACH_END_MARGIN = 0.005 # 端部の把持除外幅 [m]

# 干渉判定の設定
CONTACT_THRESHOLD = 0.5   # [N]
JOINT_LIMIT_MARGIN = 0.03 # [rad]

# 立てた T の設定
STAND_HANDOVER_CENTER = (0.0, -0.05, 0.055)  # 青の受け渡し位置
STAND_APPROACH = 0.04     # 上端把持前の待機距離 [m]
STAND_WAIT_DZ = 0.01      # 待機時の追加高さ [m]
END_GRASP_DEPTH = 0.015   # 上端把持時の下げ量 [m]
END_GRASP_LATERAL = 0.012 # 上端把持時の可動爪側へのずれ [m]
STAND_XY = (0.12, -0.14)  # 立てた T の位置
STAND_HOVER = 0.035       # 立てる前の浮かせ量 [m]
STAND_RETREAT = 0.035     # 配置後の上昇量 [m]
BAR_ON_STAND_HOVER = 0.015  # 赤を載せる前の浮かせ量 [m]
STAND_STEM_CENTER = (STAND_XY[0], STAND_XY[1], BLOCK_SIZE[0] / 2)
STAND_BAR_CENTER = (STAND_XY[0], STAND_XY[1], BLOCK_SIZE[0] + BLOCK_HALF_HEIGHT)
STAND_BAR_DIR = (1.0, 0.0, 0.0)

# 実機寄りアクチュエータ用の IK 積算補正
REAL_IK_I_LIMIT = 0.4       # [rad]
REAL_IK_I_GAIN = 0.02
REAL_IK_I_DECAY = 0.9995
REAL_SETTLE_MAX_TIME = 3.0  # [s]

# 手首反転リトライの TCP 誤差しきい値 [m]
RETRY_TCP_ERROR = 0.015
# wrist_roll の最小余裕 [rad]
ROLL_MARGIN_MIN = 0.15

# 配置後の退避高さ [m]
SAFE_RETREAT_Z = 0.10
# 退避時の shoulder_lift 戻し量 [rad]
SWING_LIFT = 0.3

# 工程末の静止時間 [s]
DWELL_TIME = 0.3
# 収束判定の設定
SETTLE_TOL = 0.004         # [m]
SETTLE_TOL_COARSE = 0.015  # [m]
SETTLE_MAX_TIME = 1.5      # [s]

# 物理ステップを設定
PHYSICS_DT = 1 / 120


# ----------------------------------------------------------------------
# D435i とマウント
# ----------------------------------------------------------------------

# マウントの USD とカメラ座標を読み込み
D435I_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "d435i_mount")
D435I_MOUNT_USD = os.path.join(D435I_DIR, "d435i_desk_mount_side.usda")
with open(os.path.join(D435I_DIR, "frames_side_25deg_200mm.json")) as _f:
    _D435I_FRAMES = json.load(_f)["frames"]

# マウント台座の原点を設定
D435I_MOUNT_POS = (-0.40, 0.0, 0.0)

# カラーカメラの位置と姿勢を計算（ROS 規約）
D435I_COLOR_POS = tuple(
    m + c for m, c in zip(D435I_MOUNT_POS, _D435I_FRAMES["color_optical_in_mount_m"])
)
D435I_COLOR_ROT_ROS = tuple(_D435I_FRAMES["color_optical_quat_wxyz_ros"])

# カラーカメラの解像度と内部パラメータを設定
_D435I_K = _D435I_FRAMES["intrinsics_1280x720_nominal"]
D435I_WIDTH = _D435I_K["width"]
D435I_HEIGHT = _D435I_K["height"]
D435I_INTRINSIC = [_D435I_K["fx"], 0.0, _D435I_K["cx"], 0.0, _D435I_K["fy"], _D435I_K["cy"], 0.0, 0.0, 1.0]
D435I_CLIP = (0.05, 3.0)


def make_d435i_camera_cfg(depth: bool = False) -> CameraCfg:
    # D435i のカメラ設定を作成
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/D435iColor",
        update_period=0.0,
        width=D435I_WIDTH,
        height=D435I_HEIGHT,
        data_types=["rgb", "distance_to_image_plane"] if depth else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=D435I_INTRINSIC,
            width=D435I_WIDTH,
            height=D435I_HEIGHT,
            clipping_range=D435I_CLIP,
        ),
        offset=CameraCfg.OffsetCfg(pos=D435I_COLOR_POS, rot=D435I_COLOR_ROT_ROS, convention="ros"),
    )


def make_block_cfg(prim_path: str, pos: tuple, color: tuple) -> RigidObjectCfg:
    # ブロックの設定を作成
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=BLOCK_START_QUAT),
    )


@configclass
class TTaskSceneCfg(InteractiveSceneCfg):
    # 床と照明を配置
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )

    # ロボットを配置
    feeder: ArticulationCfg = make_so101_cfg(
        prim_path="{ENV_REGEX_NS}/Feeder",
        pos=FEEDER_BASE,
        yaw_deg=-90.0,
        init_joint_pos=FEEDER_REST_POSE,
        contact_sensors=True,
    )
    builder: ArticulationCfg = make_so101_cfg(
        prim_path="{ENV_REGEX_NS}/Builder",
        pos=BUILDER_BASE,
        yaw_deg=90.0,
        init_joint_pos=REST_POSE,
        contact_sensors=True,
    )

    # ブロックを配置
    blue = make_block_cfg("{ENV_REGEX_NS}/BlueBlock", BLUE_START, (0.1, 0.3, 0.9))
    red = make_block_cfg("{ENV_REGEX_NS}/RedBlock", RED_START, (0.9, 0.15, 0.1))

    # 腕同士の接触を検出
    arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Feeder/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Builder/.*"],
        update_period=0.0,
        history_length=1,
    )

    # D435i のマウントを配置
    d435i_mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/D435iMount",
        spawn=sim_utils.UsdFileCfg(usd_path=D435I_MOUNT_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=D435I_MOUNT_POS),
    )

    # D435i のカメラ
    d435i_color: CameraCfg | None = None

    # 腕とブロックの接触を検出
    feeder_block_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Feeder/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BlueBlock", "{ENV_REGEX_NS}/RedBlock"],
        update_period=0.0,
        history_length=1,
    )
    builder_block_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Builder/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BlueBlock", "{ENV_REGEX_NS}/RedBlock"],
        update_period=0.0,
        history_length=1,
    )


# 接触センサの列順
BLOCK_CONTACT_ORDER = ("青", "赤")


# ----------------------------------------------------------------------
# D435i によるブロック認識
# ----------------------------------------------------------------------

# 深度の設定
DEPTH_NOISE_REL = 0.005                # 距離の 0.5 %
D435I_DEPTH_MIN = 0.28                 # 最短測距距離 [m]

# 検出クエリを設定
BLOCK_QUERIES = {"青": "a blue block", "赤": "a red block"}

# 上面とみなす高さ範囲 [m]
TOP_Z_RANGE = (0.025, 0.045)
# 遮蔽判定の長さ許容量 [m]
OCCLUSION_TOL = 0.006

# 初期位置のランダム化範囲
RANDOM_XY = 0.02          # [m]（±）
RANDOM_YAW_DEG = 25.0     # [deg]（±）
