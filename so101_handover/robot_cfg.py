import dataclasses
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# リポジトリのルートを取得
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# URDF のパスを設定
SO101_URDF_PATH = os.environ.get(
    "SO101_URDF_PATH",
    os.path.join(REPO_ROOT, "assets", "so101", "so101_new_calib.urdf"),
)

# 関節名を定義
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_JOINT_NAMES = JOINT_NAMES[:5]
GRIPPER_JOINT_NAME = "gripper"

# 手先のリンク名を定義
HAND_BODY_NAME = "gripper_link"


def yaw_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    # ヨー角をクォータニオンに変換
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# 公式 MuJoCo モデル (so101_new_calib.xml) の STS3215 の値を定義
MJCF_PARAMS = {
    "kp": 998.22,          # [N·m/rad]
    "kv": 2.731,           # [N·m·s/rad]
    "damping": 0.60,       # [N·m·s/rad]
    "frictionloss": 0.052, # [N·m]
    "armature": 0.028,     # [kg·m²]
    "forcerange": 3.35,    # [N·m]
}

# 負荷時の応答の概算値を定義
# stiffness [N·m/rad]、damping [N·m·s/rad]、delay_ms [ms]、armature [kg·m²]、friction [N·m]
REAL_ACTUATOR_PARAMS = {
    "shoulder_pan":  {"stiffness": 30.0,  "damping": 3.0,  "delay_ms": 50, "armature": 0.05},
    "shoulder_lift": {"stiffness": 15.0,  "damping": 3.0,  "delay_ms": 0,  "armature": 0.15, "friction": 0.6},
    "elbow_flex":    {"stiffness": 12.0,  "damping": 2.0,  "delay_ms": 0,  "armature": 0.10, "friction": 0.5},
    "wrist_flex":    {"stiffness": 12.0,  "damping": 1.0,  "delay_ms": 60, "armature": 0.02, "friction": 0.4},
    "wrist_roll":    {"stiffness": 100.0, "damping": 7.0,  "delay_ms": 60, "armature": 0.05},
    "gripper":       {"stiffness": 3.0,   "damping": 0.25, "delay_ms": 60, "armature": 0.006},
}
REAL_EFFORT_LIMIT = 2.5
PHYSICS_DT = 1.0 / 120.0


def ideal_actuators() -> dict:
    # 理想アクチュエータを作成
    return {
        # 腕の関節を設定
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness={
                "shoulder_pan": 150.0,
                "shoulder_lift": 150.0,
                "elbow_flex": 100.0,
                "wrist_flex": 60.0,
                "wrist_roll": 40.0,
            },
            damping={
                "shoulder_pan": 40.0,
                "shoulder_lift": 40.0,
                "elbow_flex": 30.0,
                "wrist_flex": 15.0,
                "wrist_roll": 10.0,
            },
            effort_limit_sim=2.5,
            velocity_limit_sim=2.0,
        ),
        # グリッパを設定
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME],
            stiffness=40.0,
            damping=10.0,
            effort_limit_sim=2.5,
            velocity_limit_sim=2.0,
        ),
    }


def _cfg_fields(cfg_class) -> set[str]:
    # 設定クラスの項目名を取得
    return {f.name for f in dataclasses.fields(cfg_class)}


def mjcf_actuators() -> dict:
    # MuJoCo モデルの値でアクチュエータを作成
    p = MJCF_PARAMS
    kwargs = {
        "joint_names_expr": JOINT_NAMES,
        "stiffness": p["kp"],
        "damping": p["kv"] + p["damping"],
        "armature": p["armature"],
        "effort_limit_sim": p["forcerange"],
    }
    if "friction" in _cfg_fields(ImplicitActuatorCfg):
        kwargs["friction"] = p["frictionloss"]
    return {"all": ImplicitActuatorCfg(**kwargs)}


def real_actuators() -> dict:
    # 概算値で関節ごとのアクチュエータを作成
    names = _cfg_fields(DelayedPDActuatorCfg)
    actuators = {}
    for joint, p in REAL_ACTUATOR_PARAMS.items():
        delay = round(p["delay_ms"] / 1000.0 / PHYSICS_DT)
        kwargs = {
            "joint_names_expr": [joint],
            "stiffness": p["stiffness"],
            "damping": p["damping"],
            "armature": p["armature"],
            "min_delay": delay,
            "max_delay": delay,
            "effort_limit": REAL_EFFORT_LIMIT,
        }
        if "friction" in p and "friction" in names:
            kwargs["friction"] = p["friction"]
        actuators[joint] = DelayedPDActuatorCfg(**kwargs)
    return actuators


def make_actuators(profile: str) -> dict:
    # 名前からアクチュエータを選択
    if profile == "mjcf":
        return mjcf_actuators()
    if profile == "real":
        return real_actuators()
    return ideal_actuators()


def make_so101_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    yaw_deg: float,
    init_joint_pos: dict[str, float],
    contact_sensors: bool = False,
    actuator: str = "ideal",
) -> ArticulationCfg:
    # SO-101 の設定を作成
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=SO101_URDF_PATH,
            fix_base=True,
            merge_fixed_joints=True,
            collider_type="convex_decomposition",
            activate_contact_sensors=contact_sensors,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=yaw_to_quat(yaw_deg),
            joint_pos=init_joint_pos,
        ),
        actuators=make_actuators(actuator),
        soft_joint_pos_limit_factor=0.95,
    )
