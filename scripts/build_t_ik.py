"""IK と状態機械で T 字を組み立てる。"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=1.0, help="動作速度の倍率")
parser.add_argument("--hold", action="store_true", help="終了後もウィンドウを保持")
parser.add_argument("--check_ready", action="store_true", help="休息姿勢で停止")
parser.add_argument("--continue_on_fail", action="store_true", help="把持失敗時も続行")
parser.add_argument("--perception", action="store_true", help="D435i 画像と物体検出でブロック位置を推定")
parser.add_argument("--randomize", action="store_true", help="ブロック初期姿勢をランダム化")
parser.add_argument("--seed", type=int, default=0, help="乱数シード")
parser.add_argument("--det_threshold", type=float, default=0.1, help="検出スコアの閾値")
parser.add_argument("--image_dir", type=str, default="outputs/perception", help="検出画像の保存先")
parser.add_argument("--depth_noise", action="store_true", help="距離比例の深度ノイズを付加")
parser.add_argument("--actuator", choices=["ideal", "mjcf", "real"], default="ideal",
                    help="ideal: 理想、mjcf: MuJoCo 公式値、real: 実機想定の概算値")
parser.add_argument("--shape", choices=["standing", "flat"], default="standing",
                    help="standing: 青を立てた T、flat: 平置きの T")
parser.add_argument("--side_camera", action="store_true", help="D435i カラー画像を連番 JPEG で保存")
parser.add_argument("--side_dir", type=str, default="outputs/side_camera", help="D435i 画像の保存先")
parser.add_argument("--side_fps", type=float, default=30.0, help="D435i 画像の保存レート（シミュレーション時間基準）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.perception or args.side_camera:
    args.enable_cameras = True

# Isaac Sim を起動
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import PhysxCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from so101_handover import t_task_cfg as T
from so101_handover.robot_cfg import SO101_URDF_PATH, make_actuators
from so101_handover.perception import (
    BlockDetector,
    estimate_block_pose,
    save_detection_image,
    yaw_to_quat,
)
from so101_handover.ik import (
    TCP_OFFSET,
    DlsIk,
    approach_dir,
    horizontal_grasp_quat,
    horizontal_unit,
    pitch_to_approach,
    tcp_axis,
    top_down_quat,
)

Pose = tuple[torch.Tensor, torch.Tensor]


# ----------------------------------------------------------------------
# ブロックの姿勢
# ----------------------------------------------------------------------


def block_pose(block) -> Pose:
    # ブロックの位置と姿勢を取得
    return block.data.root_pos_w.clone(), block.data.root_quat_w.clone()


def block_axis(quat: torch.Tensor) -> torch.Tensor:
    # ブロックの長手方向を取得
    x = torch.tensor([[1.0, 0.0, 0.0]], device=quat.device)
    return quat_apply(quat, x)


def block_ends(block) -> tuple[torch.Tensor, torch.Tensor]:
    # ブロック両端の把持点を取得
    pos, quat = block_pose(block)
    axis = block_axis(quat)
    return pos + axis * T.GRIP_OFFSET, pos - axis * T.GRIP_OFFSET


def align_quat(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src を dst に重ねる最小回転を計算
    a = src / src.norm(dim=-1, keepdim=True)
    b = dst / dst.norm(dim=-1, keepdim=True)
    axis = torch.linalg.cross(a, b, dim=-1)
    sin = axis.norm(dim=-1)
    cos = (a * b).sum(dim=-1)
    if sin.item() < 1e-6:
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=src.device)
    angle = torch.atan2(sin, cos)
    return quat_from_angle_axis(angle, axis / sin.unsqueeze(-1))


def grasp_offsets(tcp_pos: torch.Tensor, tcp_quat: torch.Tensor, block) -> tuple[float, float, float]:
    # TCP からブロック中心線までの横・深さ・幅方向のずれを計算
    pos, quat = block_pose(block)
    axis = block_axis(quat)
    half = T.BLOCK_SIZE[0] / 2 - T.ATTACH_END_MARGIN
    t = ((tcp_pos - pos) * axis).sum(dim=-1, keepdim=True).clamp(-half, half)
    v = pos + axis * t - tcp_pos
    lateral = (v * tcp_axis(tcp_quat, (1.0, 0.0, 0.0))).sum().item()
    depth = (v * approach_dir(tcp_quat)).sum().item()
    width = (v * tcp_axis(tcp_quat, (0.0, 1.0, 0.0))).sum().item()
    return lateral, depth, width


def jaw_gap(open_angle: float) -> float:
    # グリッパ角度から爪の隙間を計算
    return 0.016 + 0.076 * open_angle


def can_grasp(lateral: float, depth: float, width: float, open_angle: float) -> bool:
    # ずれが把持可能範囲内かを判定
    half = T.BLOCK_SIZE[1] / 2
    lo = T.FIXED_JAW_X + half - T.ATTACH_PENETRATION
    hi = T.FIXED_JAW_X + jaw_gap(open_angle) - half + T.ATTACH_PENETRATION
    in_lateral = lo <= lateral <= hi
    in_depth = T.ATTACH_DEPTH_MIN <= depth <= T.ATTACH_DEPTH_MAX
    in_width = abs(width) <= T.ATTACH_WIDTH
    return in_lateral and in_depth and in_width


# ----------------------------------------------------------------------
# 腕の制御
# ----------------------------------------------------------------------


class ArmController:
    def __init__(self, name: str, robot, base_pos: torch.Tensor, rest_pose: dict[str, float]):
        self.name = name
        self.rest_pose = rest_pose
        self.robot = robot
        self.ik = DlsIk(robot)
        self.base_pos = base_pos
        dev = base_pos.device

        # 休息姿勢の関節角を作成
        self.rest_arm = torch.tensor(
            [[rest_pose[robot.joint_names[i]] for i in self.ik.arm_ids]], device=dev
        )
        self.grip = rest_pose["gripper"]

        # 指令を初期化
        zero_pos = torch.zeros(1, 3, device=dev)
        unit_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        self.cmd_pos, self.cmd_quat = zero_pos, unit_quat
        self.start: Pose = (zero_pos, unit_quat)
        self.goal: Pose = (zero_pos, unit_quat)

        # 把持状態を初期化
        self.held = None
        self.held_rel: Pose | None = None
        self.rel_start: Pose | None = None
        self.rel_goal: Pose | None = None

        # 関節インデックスを取得
        self.roll_id = robot.find_joints(["wrist_roll"])[0][0]
        self.pan_id = robot.find_joints(["shoulder_pan"])[0][0]
        self.pan_col = list(self.ik.arm_ids).index(self.pan_id)
        self.lift_col = list(self.ik.arm_ids).index(robot.find_joints(["shoulder_lift"])[0][0])

        # 関節直接駆動の目標を初期化
        self.joint_target: torch.Tensor | None = None
        self.home_start: torch.Tensor | None = None
        self.home_goal: torch.Tensor | None = None

        # 手首候補の選択状態を初期化
        self.flip_next = False
        self.last_choice: int | None = None

    # 爪の隙間の中心 ---------------------------------------------------

    def setup_grasp_center(self):
        # TCP を爪の隙間の中心に設定
        jaw_ids = self.robot.find_bodies(".*jaw.*")[0]
        if not jaw_ids:
            print(f"[jaw] {self.name}: 可動爪リンクなし")
            return
        hand_pos = self.robot.data.body_pos_w[:, self.ik.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.ik.hand_id]
        jaw_pos = self.robot.data.body_pos_w[:, jaw_ids[0]]
        jaw_local, _ = subtract_frame_transforms(hand_pos, hand_quat, jaw_pos)
        sign = 1.0 if jaw_local[0, 0].item() >= 0 else -1.0
        offset = (sign * T.GRASP_LATERAL, TCP_OFFSET[1], TCP_OFFSET[2])
        self.ik.set_tcp_offset(offset)
        local = [round(v * 1000, 1) for v in jaw_local[0].tolist()]
        print(f"[jaw] {self.name}: {self.robot.body_names[jaw_ids[0]]} {local} mm、"
              f"TCP X {sign * T.GRASP_LATERAL * 1000:+.0f} mm")

    # 指令 -------------------------------------------------------------

    def capture(self):
        # 現在の TCP を指令に設定
        pos, quat = self.ik.tcp_pose()
        self.cmd_pos, self.cmd_quat = pos.clone(), quat.clone()
        self.start = (pos.clone(), quat.clone())
        self.goal = (pos.clone(), quat.clone())

    def set_goal(self, goal: Pose):
        # 目標への移動を開始
        self.start = (self.cmd_pos.clone(), self.cmd_quat.clone())
        self.goal = (goal[0].clone(), goal[1].clone())

    def interpolate(self, alpha: float):
        # 指令を補間して更新
        p0, q0 = self.start
        p1, q1 = self.goal
        if (q0 * q1).sum() < 0:
            q1 = -q1
        self.cmd_pos = p0 + (p1 - p0) * alpha
        q = q0 + (q1 - q0) * alpha
        self.cmd_quat = q / q.norm(dim=-1, keepdim=True)

    def apply(self):
        # 関節目標を送信
        if self.joint_target is not None:
            self.robot.set_joint_position_target(self.joint_target, joint_ids=self.ik.arm_ids)
        else:
            q_arm = self.ik.solve(self.cmd_pos, self.cmd_quat)
            self.robot.set_joint_position_target(q_arm, joint_ids=self.ik.arm_ids)
        self.apply_grip()

    def apply_rest_joints(self):
        # 休息姿勢の関節角を送信
        self.robot.set_joint_position_target(self.rest_arm, joint_ids=self.ik.arm_ids)
        self.apply_grip()

    def apply_grip(self):
        # グリッパ角度を送信
        grip = torch.full((1, 1), self.grip, device=self.base_pos.device)
        self.robot.set_joint_position_target(grip, joint_ids=[self.ik.grip_id])

    # 把持 -------------------------------------------------------------

    def try_attach(self, blocks: dict) -> bool:
        # 爪の間のブロックを手先に固定
        tcp_pos, tcp_quat = self.ik.tcp_pose()
        best_name, best_score, best_off = None, 1e9, (0.0, 0.0, 0.0)
        for name, block in blocks.items():
            lateral, depth, width = grasp_offsets(tcp_pos, tcp_quat, block)
            score = abs(lateral) + abs(depth) + abs(width)
            if score < best_score:
                best_name, best_score, best_off = name, score, (lateral, depth, width)

        lateral, depth, width = best_off
        detail = f"横 {lateral * 1000:+.1f} mm、深さ {depth * 1000:+.1f} mm、幅 {width * 1000:+.1f} mm"
        if not can_grasp(lateral, depth, width, self.grip):
            print(f"        [把持失敗] {self.name}: {best_name}（{detail}）")
            return False

        # 閉じる前の相対姿勢を記録
        block = blocks[best_name]
        pos, quat = block_pose(block)
        rel_pos, rel_quat = subtract_frame_transforms(tcp_pos, tcp_quat, pos, quat)
        self.held = block
        self.held_rel = (rel_pos, rel_quat)
        self.rel_start = (rel_pos.clone(), rel_quat.clone())

        # 閉じ終えたときの相対姿勢を計算
        bx = quat_apply(rel_quat, torch.tensor([[1.0, 0.0, 0.0]], device=pos.device))
        k = 1 if abs(bx[0, 1].item()) >= abs(bx[0, 2].item()) else 2
        target = torch.zeros(1, 3, device=pos.device)
        target[0, k] = 1.0 if bx[0, k].item() >= 0 else -1.0
        goal_quat = quat_mul(align_quat(bx, target), rel_quat)
        goal_pos = rel_pos.clone()
        goal_pos[0, 0] = T.GRASP_CENTER_X
        self.rel_goal = (goal_pos, goal_quat)

        print(f"        [把持] {self.name}: {best_name}（{detail}）")
        return True

    def update_grasp(self, alpha: float):
        # ブロックを爪の中心へ移動
        if self.held is None or self.rel_start is None:
            return
        p0, q0 = self.rel_start
        p1, q1 = self.rel_goal
        if (q0 * q1).sum() < 0:
            q1 = -q1
        q = q0 + (q1 - q0) * alpha
        self.held_rel = (p0 + (p1 - p0) * alpha, q / q.norm(dim=-1, keepdim=True))

    def release(self):
        # 固定を解除
        self.held = None
        self.held_rel = None
        self.rel_start = None
        self.rel_goal = None

    def apply_attachment(self):
        # 固定中のブロックを手先に追従
        if self.held is None:
            return
        tcp_pos, tcp_quat = self.ik.tcp_pose()
        pos, quat = combine_frame_transforms(tcp_pos, tcp_quat, self.held_rel[0], self.held_rel[1])
        self.held.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        self.held.write_root_velocity_to_sim(torch.zeros(1, 6, device=pos.device))

    # 手首の向きの選択 ---------------------------------------------------

    def roll_prediction(self, goal: Pose) -> tuple[float, float]:
        # wrist_roll の回転量と可動域の余裕を予測
        tcp_pos, tcp_quat = self.ik.tcp_pose()
        data = self.robot.data
        limits = getattr(data, "joint_pos_limits", data.soft_joint_pos_limits)[0, self.roll_id]
        roll = data.joint_pos[0, self.roll_id].item()

        # 手先 Y 軸の水平回転量を計算
        y_now = horizontal_unit(tcp_axis(tcp_quat, (0.0, 1.0, 0.0)))[0]
        y_goal = horizontal_unit(tcp_axis(goal[1], (0.0, 1.0, 0.0)))[0]
        d_yaw = math.atan2((y_now[0] * y_goal[1] - y_now[1] * y_goal[0]).item(), (y_now * y_goal).sum().item())

        # shoulder_pan による方位変化を除外
        rel_now = (tcp_pos - self.base_pos)[0]
        rel_goal = (goal[0] - self.base_pos)[0]
        az_now = math.atan2(rel_now[1].item(), rel_now[0].item())
        az_goal = math.atan2(rel_goal[1].item(), rel_goal[0].item())
        d_roll_yaw = math.remainder(d_yaw - math.remainder(az_goal - az_now, 2 * math.pi), 2 * math.pi)

        # ヤコビアンから回転の向きを取得
        jac = self.robot.root_physx_view.get_jacobians()[0, self.ik.jac_id, 5, self.roll_id].item()
        d_roll = d_roll_yaw if jac >= 0 else -d_roll_yaw

        roll_new = roll + d_roll
        margin = min(roll_new - limits[0].item(), limits[1].item() - roll_new)
        return d_roll, margin

    def choose_top_down(self, candidates: list[Pose]) -> Pose:
        # 手首の回転量が小さい候補を選択
        preds = [self.roll_prediction(c) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: abs(preds[i][0]))
        idx = next((i for i in order if preds[i][1] >= T.ROLL_MARGIN_MIN), max(order, key=lambda i: preds[i][1]))
        if self.flip_next and len(candidates) == 2 and self.last_choice is not None:
            idx = 1 - self.last_choice
            self.flip_next = False
        self.last_choice = idx
        detail = " / ".join(f"回転 {d:+.2f} 余裕 {m:+.2f}" for d, m in preds)
        print(f"        [手首] {self.name}: {detail} rad → 候補 {idx}")
        return candidates[idx]

    def roll_stuck(self) -> bool:
        # wrist_roll の可動域端での停止を判定
        at_limit = any(name.startswith("wrist_roll") for name in self.joints_at_limit())
        return at_limit and self.tcp_error() > T.RETRY_TCP_ERROR

    # 状態の表示 -------------------------------------------------------

    def tcp_error(self) -> float:
        # 指令と実際の TCP の距離を計算
        pos, _ = self.ik.tcp_pose()
        return (pos - self.cmd_pos).norm().item()

    def downward(self) -> float:
        # 手先の下向き度を計算
        _, quat = self.ik.tcp_pose()
        return -approach_dir(quat)[0, 2].item()

    def joints_at_limit(self) -> list[str]:
        # 可動域の端にある関節を取得
        data = self.robot.data
        limits = getattr(data, "joint_pos_limits", data.soft_joint_pos_limits)[0]
        q = data.joint_pos[0]
        out = []
        for i, name in enumerate(self.robot.joint_names):
            if name == "gripper":
                continue
            if q[i] < limits[i, 0] + T.JOINT_LIMIT_MARGIN:
                out.append(f"{name}(下限)")
            elif q[i] > limits[i, 1] - T.JOINT_LIMIT_MARGIN:
                out.append(f"{name}(上限)")
        return out


# ----------------------------------------------------------------------
# 干渉の検出
# ----------------------------------------------------------------------


class ContactMonitor:
    # 腕どうし・腕とブロックの接触を記録

    def __init__(self, arm_sensor, builder_robot, block_sensors: dict, arms: dict, blocks: dict):
        self.arm_sensor = arm_sensor
        self.feeder_names = arm_sensor.body_names
        self.builder_names = builder_robot.body_names
        self.block_sensors = block_sensors
        self.arms = arms
        self.block_list = [(name, blocks[name]) for name in T.BLOCK_CONTACT_ORDER]
        self.skip: set[str] = set()
        self.reset()

    def reset(self):
        # 工程ごとの最大値を初期化
        self.arm_max = 0.0
        self.arm_pair = None
        self.block_max = {key: (0.0, "", "") for key in self.block_sensors}

    def update(self):
        # 腕間の接触力の最大値を記録
        forces = self.arm_sensor.data.force_matrix_w
        if forces is not None:
            mag = forces[0].norm(dim=-1)
            value, index = mag.flatten().max(dim=0)
            if value.item() > self.arm_max:
                self.arm_max = value.item()
                self.arm_pair = divmod(index.item(), mag.shape[1])

        # 腕と非把持ブロックの接触力の最大値を記録
        for key, sensor in self.block_sensors.items():
            if key in self.skip:
                continue
            forces = sensor.data.force_matrix_w
            if forces is None:
                continue
            mag = forces[0].norm(dim=-1)
            for j, (name, block) in enumerate(self.block_list):
                if self.arms[key].held is block:
                    continue
                value, i = mag[:, j].max(dim=0)
                if value.item() > self.block_max[key][0]:
                    self.block_max[key] = (value.item(), sensor.body_names[i.item()], name)

    def summary(self) -> str:
        # 工程中の干渉を集計
        out = []
        if self.arm_max >= T.CONTACT_THRESHOLD and self.arm_pair is not None:
            i, j = self.arm_pair
            a = self.feeder_names[i] if i < len(self.feeder_names) else str(i)
            b = self.builder_names[j] if j < len(self.builder_names) else str(j)
            out.append(f"feeder/{a} と builder/{b}  最大 {self.arm_max:.1f} N")
        for key, (value, body, name) in self.block_max.items():
            if value >= T.CONTACT_THRESHOLD:
                out.append(f"{key}/{body} と {name}  最大 {value:.1f} N")
        return "あり  " + "、".join(out) if out else "なし"


# ----------------------------------------------------------------------
# 目標の計算
# ----------------------------------------------------------------------


def lifted(pose: Pose, height: float) -> Pose:
    # 姿勢を保ったまま高さを変更
    pos = pose[0].clone()
    pos[..., 2] += height
    return pos, pose[1].clone()


def top_down_grasp(arm: ArmController, pose: Pose) -> Pose:
    # 土台に近い端を真上から掴む姿勢を計算
    pos, quat = pose
    axis = horizontal_unit(block_axis(quat))
    ends = (pos + axis * T.GRIP_OFFSET, pos - axis * T.GRIP_OFFSET)
    dists = [(e - arm.base_pos)[..., :2].norm().item() for e in ends]
    grip = ends[0 if dists[0] < dists[1] else 1].clone()
    grip[..., 2] = pos[..., 2] + T.GRASP_DZ

    # 手首の向きを選択
    return arm.choose_top_down([(grip, top_down_quat(axis)), (grip, top_down_quat(-axis))])


def horizontal_grasp(arm: ArmController, block, approach: torch.Tensor, other_tcp: torch.Tensor, back: float) -> Pose:
    # 相手の TCP から遠い端を水平に掴む姿勢を計算
    ends = block_ends(block)
    dists = [(e - other_tcp).norm().item() for e in ends]
    grip = ends[0 if dists[0] > dists[1] else 1]
    a = horizontal_unit(approach)
    quat = horizontal_grasp_quat(a, arm.cmd_quat)
    # 可動爪側へオフセット
    x_axis = tcp_axis(quat, (1.0, 0.0, 0.0))
    return grip - a * back - x_axis * T.HORIZONTAL_GRASP_LATERAL, quat


def carry_top_down(arm: ArmController, center: torch.Tensor, direction: torch.Tensor) -> Pose:
    # ブロックを center・direction に置く TCP 姿勢を計算
    rel_pos, rel_quat = arm.held_rel

    # TCP から見た長手方向の角度を取得
    bx = quat_apply(rel_quat, torch.tensor([[1.0, 0.0, 0.0]], device=rel_quat.device))[0]
    phi = math.atan2(bx[1].item(), bx[0].item())

    theta = math.atan2(direction[0, 1].item(), direction[0, 0].item())
    candidates = []
    for block_yaw in (theta, theta + math.pi):
        # 真下向きの候補を作成
        psi = block_yaw - phi
        y_axis = torch.tensor([[-math.sin(psi), math.cos(psi), 0.0]], device=center.device)
        tcp_quat = top_down_quat(y_axis)
        tcp_pos = center - quat_apply(tcp_quat, rel_pos)
        candidates.append((tcp_pos, tcp_quat))
    return arm.choose_top_down(candidates)


def end_grasp(arm: ArmController, block, toward: torch.Tensor, back: float) -> Pose:
    # 縦のブロックの上端をつまむ姿勢を計算
    pos, quat = block_pose(block)
    axis = block_axis(quat)
    if axis[0, 2].item() < 0:
        axis = -axis
    return end_grasp_at(arm, pos + axis * (T.BLOCK_SIZE[0] / 2), toward, back)


def end_grasp_at(arm: ArmController, top: torch.Tensor, toward: torch.Tensor, back: float) -> Pose:
    # 上面中心 top のブロックをつまむ姿勢を計算
    grip = top.clone()
    grip[..., 2] -= T.END_GRASP_DEPTH
    t = horizontal_unit(toward)
    candidates = []
    for d in (t, -t):
        q = top_down_quat(d)
        # 可動爪側へオフセット
        x_axis = tcp_axis(q, (1.0, 0.0, 0.0))
        candidates.append((grip - x_axis * T.END_GRASP_LATERAL - t * back, q))
    return arm.choose_top_down(candidates)


def carry_held_top_down(arm: ArmController, center: torch.Tensor, y_dir: torch.Tensor) -> Pose:
    # ブロックを center に置く TCP 姿勢を計算
    rel_pos, _ = arm.held_rel
    candidates = []
    for d in (y_dir, -y_dir):
        q = top_down_quat(d)
        candidates.append((center - quat_apply(q, rel_pos), q))
    return arm.choose_top_down(candidates)


# ----------------------------------------------------------------------
# 手順の定義
# ----------------------------------------------------------------------


@dataclass
class Step:
    name: str
    duration: float
    feeder: Callable[[], Pose] | None = None
    builder: Callable[[], Pose] | None = None
    feeder_grip: str | None = None   # "open"、"open_wide"、"close"
    builder_grip: str | None = None
    action: Callable[[], bool] | None = None  # 工程末の処理
    retry_flip: bool = False  # 手首反転で再試行
    home: tuple[str, ...] = ()  # 関節直接駆動の腕
    home_to: str = "rest"  # "rest" または "pan"
    precise: bool = False  # SETTLE_TOL まで収束待ち
    coarse: tuple[str, ...] = ()  # SETTLE_TOL_COARSE で待つ腕


def make_builder_return_steps(builder, label) -> list[Step]:
    # 組立側の退避手順を作成
    def up():
        pos = builder.cmd_pos.clone()
        pos[..., 2] = max(pos[0, 2].item(), builder.base_pos[0, 2].item() + T.SAFE_RETREAT_Z)
        return pos, builder.cmd_quat.clone()

    return [
        Step(f"{label}: 組立側が真上に上がる", 0.8, builder=up),
        Step(f"{label}: 組立側が上がりながら T 字から離れる", 1.0, home=("builder",), home_to="pan"),
        Step(f"{label}: 組立側が休息姿勢へ", 1.5, home=("builder",), home_to="rest"),
    ]


def make_pick_steps(feeder, label, pick_pose: Callable[[], Pose], perceive: Callable[[], bool] | None) -> list[Step]:
    # 供給側の把持手順を作成
    H = T.HOVER_HEIGHT
    steps = []
    if perceive is not None:
        # ブロック位置を推定
        steps.append(Step(f"{label}: カメラでブロックを検出", 0.3, action=perceive))
    return steps + [
        Step(f"{label}: 供給側が掴む位置の上へ", 1.5,
             feeder=lambda: lifted(top_down_grasp(feeder, pick_pose()), H), feeder_grip="open",
             retry_flip=True),
        Step(f"{label}: 供給側が降下", 1.0, feeder=lambda: lifted((feeder.cmd_pos, feeder.cmd_quat), -H), precise=True),
        Step(f"{label}: 供給側が把持", 0.6, feeder_grip="close", precise=True),
        Step(f"{label}: 供給側が持ち上げ", 0.8, feeder=lambda: lifted((feeder.cmd_pos, feeder.cmd_quat), H)),
    ]


def make_standing_stem_steps(
    feeder, builder, block, label, low_center, stand_center,
    pick_pose: Callable[[], Pose], perceive: Callable[[], bool] | None = None,
) -> list[Step]:
    # 青を立てて置く手順を作成
    D = T.STAND_APPROACH
    to_builder = horizontal_unit(builder.base_pos - feeder.base_pos)
    to_feeder = -to_builder

    def cmd(arm):
        return arm.cmd_pos, arm.cmd_quat

    def feeder_low_handover():
        # 低い受け渡し位置の TCP 姿勢を計算
        quat = pitch_to_approach(feeder.cmd_quat, to_builder)
        rel_pos, _ = feeder.held_rel
        return low_center - quat_apply(quat, rel_pos), quat

    def builder_end_grasp(back: float):
        # 上端をつまむ姿勢を計算
        return end_grasp(builder, block, to_feeder, back)

    def builder_wait():
        # 受け渡し位置の上の待機姿勢を計算
        top = low_center.clone()
        top[..., 2] += T.BLOCK_SIZE[0] / 2
        return lifted(end_grasp_at(builder, top, to_feeder, D), T.STAND_WAIT_DZ)

    def builder_y():
        return horizontal_unit(tcp_axis(builder.cmd_quat, (0.0, 1.0, 0.0)))

    def builder_stand_hover():
        center = stand_center.clone()
        center[..., 2] += T.STAND_HOVER
        return carry_held_top_down(builder, center, builder_y())

    return make_pick_steps(feeder, label, pick_pose, perceive) + [
        # 低い位置で受け渡し
        Step(f"{label}: 供給側が手首を起こして低い受け渡し位置へ、組立側が受け取りに上がる", 2.0,
             feeder=feeder_low_handover, builder=builder_wait, builder_grip="open_wide",
             retry_flip=True, precise=True, coarse=("builder",)),
        Step(f"{label}: 組立側が上端の手前へ", 1.0, builder=lambda: builder_end_grasp(D), retry_flip=True),
        Step(f"{label}: 組立側が前進", 0.8, builder=lambda: builder_end_grasp(0.0), precise=True),
        Step(f"{label}: 組立側が上端を把持", 0.6, builder_grip="close", precise=True),
        Step(f"{label}: 供給側が解放", 0.5, feeder_grip="open"),
        Step(f"{label}: 供給側が後退", 0.8,
             feeder=lambda: (feeder.cmd_pos - to_builder * D, feeder.cmd_quat.clone())),

        # 縦のまま設置
        Step(f"{label}: 供給側が休息姿勢へ、組立側が立てる位置の上へ", 2.0,
             builder=builder_stand_hover, retry_flip=True, home=("feeder",), home_to="rest"),
        Step(f"{label}: 組立側が降下して立てる", 1.2,
             builder=lambda: lifted(cmd(builder), T.RELEASE_DZ - T.STAND_HOVER), precise=True),
        Step(f"{label}: 組立側が解放", 0.5, builder_grip="open"),
        Step(f"{label}: 組立側が退避", 0.8, builder=lambda: lifted(cmd(builder), T.STAND_RETREAT)),
    ] + make_builder_return_steps(builder, label)


def make_block_steps(
    feeder, builder, block, label, handover_center, place_center, place_dir,
    pick_pose: Callable[[], Pose], perceive: Callable[[], bool] | None = None,
    place_hover: float = T.HOVER_HEIGHT, retreat: float = T.HOVER_HEIGHT,
) -> list[Step]:
    # 把持・受け渡し・設置の手順を作成
    D = T.APPROACH_DIST
    to_builder = horizontal_unit(builder.base_pos - feeder.base_pos)
    to_feeder = -to_builder
    down = torch.tensor([[0.0, 0.0, -1.0]], device=to_builder.device)

    def cmd(arm):
        return arm.cmd_pos, arm.cmd_quat

    def feeder_handover():
        # 受け渡し位置の TCP 姿勢を計算
        quat = pitch_to_approach(feeder.cmd_quat, to_builder)
        rel_pos, _ = feeder.held_rel
        return handover_center - quat_apply(quat, rel_pos), quat

    def feeder_tcp():
        return feeder.ik.tcp_pose()[0]

    def builder_reorient():
        # 手先を真下に向ける姿勢を計算
        pos = builder.base_pos + to_feeder * T.REORIENT_RADIUS
        pos[..., 2] = builder.base_pos[..., 2] + T.REORIENT_HEIGHT
        return pos, pitch_to_approach(builder.cmd_quat, down)

    return make_pick_steps(feeder, label, pick_pose, perceive) + [
        # 握手型で受け渡し
        Step(f"{label}: 供給側が手首を起こして受け渡し位置へ", 2.0, feeder=feeder_handover, precise=True),
        Step(f"{label}: 組立側が掴む端の高さで手前へ", 2.0,
             builder=lambda: horizontal_grasp(builder, block, to_feeder, feeder_tcp(), D), builder_grip="open_wide"),
        Step(f"{label}: 組立側が前進", 0.8,
             builder=lambda: horizontal_grasp(builder, block, to_feeder, feeder_tcp(), -T.HORIZONTAL_GRASP_DEPTH), precise=True),
        Step(f"{label}: 組立側が把持", 0.6, builder_grip="close", precise=True),
        Step(f"{label}: 供給側が解放", 0.5, feeder_grip="open"),
        Step(f"{label}: 供給側が後退", 0.8,
             feeder=lambda: (feeder.cmd_pos - to_builder * D, feeder.cmd_quat.clone())),

        # 向きを戻して設置
        Step(f"{label}: 供給側が休息姿勢へ、組立側が下がりながら手首を倒す", 2.0,
             builder=builder_reorient, home=("feeder",), home_to="rest"),
        Step(f"{label}: 組立側が置く位置の上へ", 1.5,
             builder=lambda: lifted(carry_top_down(builder, place_center, place_dir), place_hover),
             retry_flip=True),
        Step(f"{label}: 組立側が降下", 1.2,
             builder=lambda: lifted(cmd(builder), T.RELEASE_DZ - place_hover), precise=True),
        Step(f"{label}: 組立側が解放", 0.5, builder_grip="open"),
        Step(f"{label}: 組立側が退避", 0.8, builder=lambda: lifted(cmd(builder), retreat)),
    ] + make_builder_return_steps(builder, label)


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------


class SideRecorder:
    # D435i カラー画像を連番 JPEG で保存

    def __init__(self, camera, out_dir: str, fps: float, dt: float):
        self.camera = camera
        self.out_dir = out_dir
        self.every = max(1, round(1.0 / (fps * dt)))
        self.active = False
        self.steps = 0
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)
        # 既存の画像を削除
        for name in os.listdir(out_dir):
            if name.endswith(".jpg"):
                os.remove(os.path.join(out_dir, name))

    def update(self):
        # 一定間隔で画像を保存
        if not self.active:
            return
        self.steps += 1
        if self.steps % self.every != 0:
            return
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(self.out_dir, f"{self.count:05d}.jpg"), quality=92)
        self.count += 1


class Runner:
    def __init__(self, sim, scene, arms: dict[str, ArmController], blocks: dict, monitor: ContactMonitor):
        self.sim = sim
        self.scene = scene
        self.arms = arms
        self.blocks = blocks
        self.monitor = monitor
        self.recorder: SideRecorder | None = None
        self.dt = sim.get_physics_dt()
        self.sim_time = 0.0

    def physics_step(self, use_ik: bool = True):
        # 指令を送信して 1 ステップ実行
        for arm in self.arms.values():
            if use_ik:
                arm.apply()
            else:
                arm.apply_rest_joints()
        # 把持ブロックを手先に追従
        for arm in self.arms.values():
            arm.apply_attachment()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)
        self.monitor.update()
        if self.recorder is not None:
            self.recorder.update()
        self.sim_time += self.dt

    def settle(self, seconds: float, use_ik: bool = True):
        # 目標を保持して待機
        for _ in range(int(seconds / self.dt)):
            if not simulation_app.is_running():
                return
            self.physics_step(use_ik)

    def settle_until_converged(self, tols: dict[str, float]):
        # TCP 誤差の収束まで待機
        max_time = T.REAL_SETTLE_MAX_TIME if args.actuator == "real" else T.SETTLE_MAX_TIME
        for _ in range(int(max_time / self.dt)):
            if not simulation_app.is_running():
                return
            if all(arm.tcp_error() <= tols[key] for key, arm in self.arms.items()):
                return
            self.physics_step()

    def run(self, step: Step):
        # 目標とグリッパを設定
        print(f"[step] {step.name}")
        self.monitor.reset()
        self.monitor.skip = set()
        grip_actions = {}
        for key, target_fn, grip in (
            ("feeder", step.feeder, step.feeder_grip),
            ("builder", step.builder, step.builder_grip),
        ):
            arm = self.arms[key]
            arm.set_goal(target_fn() if target_fn is not None else (arm.cmd_pos, arm.cmd_quat))
            # 開閉時の爪の接触を除外
            if grip == "close" or (grip is not None and arm.held is not None):
                self.monitor.skip.add(key)
            if grip in ("open", "open_wide"):
                arm.release()
                arm.grip = T.GRIPPER_OPEN if grip == "open" else T.GRIPPER_OPEN_WIDE
            elif grip == "close":
                grip_actions[key] = arm

        # ブロックを固定して閉じる目標を設定
        ok = True
        for arm in grip_actions.values():
            attached = arm.try_attach(self.blocks)
            ok = attached and ok
            if not attached:
                continue
            arm.grip = T.GRIPPER_TOUCH

            # 相手が把持中なら手先を横へ移動
            held_by_other = any(o is not arm and o.held is arm.held for o in self.arms.values())
            if attached and held_by_other:
                shift = (arm.rel_start[0][0, 0] - arm.rel_goal[0][0, 0]).item()
                x_axis = tcp_axis(arm.cmd_quat, (1.0, 0.0, 0.0))
                arm.set_goal((arm.cmd_pos + x_axis * shift, arm.cmd_quat.clone()))
                print(f"        [把持] {arm.name}: 手先を {shift * 1000:+.1f} mm 横移動")

        # 関節直接駆動の始点と終点を設定
        home_arms = [self.arms[key] for key in step.home]
        for arm in home_arms:
            arm.home_start = arm.robot.data.joint_pos[:, arm.ik.arm_ids].clone()
            if step.home_to == "pan":
                # shoulder_pan を休息角へ、shoulder_lift を SWING_LIFT だけ戻す
                arm.home_goal = arm.home_start.clone()
                arm.home_goal[:, arm.pan_col] = arm.rest_arm[:, arm.pan_col]
                lift = arm.home_start[:, arm.lift_col]
                diff = arm.rest_arm[:, arm.lift_col] - lift
                arm.home_goal[:, arm.lift_col] = lift + diff.clamp(-T.SWING_LIFT, T.SWING_LIFT)
            else:
                arm.home_goal = arm.rest_arm
                arm.grip = arm.rest_pose["gripper"]

        # 補間しながら移動
        n = max(1, int(step.duration / args.speed / self.dt))
        for i in range(n):
            if not simulation_app.is_running():
                return
            a = (i + 1) / n
            alpha = a * a * (3.0 - 2.0 * a)
            for arm in self.arms.values():
                arm.interpolate(alpha)
            for arm in grip_actions.values():
                arm.update_grasp(alpha)
            for arm in home_arms:
                arm.joint_target = arm.home_start + (arm.home_goal - arm.home_start) * alpha
            self.physics_step()
        self.settle(T.DWELL_TIME / args.speed)

        # 現在の姿勢を IK 指令に設定
        for arm in home_arms:
            arm.capture()
            arm.joint_target = None
            arm.ik.reset_integral()
        tols = {
            key: T.SETTLE_TOL if step.precise and key not in step.coarse else T.SETTLE_TOL_COARSE
            for key in self.arms
        }
        self.settle_until_converged(tols)

        if step.action is not None:
            ok = step.action() and ok

        self.print_status()

        # 手首を反転して再試行
        if step.retry_flip:
            retry = {}
            for key, target_fn in (("feeder", step.feeder), ("builder", step.builder)):
                arm = self.arms[key]
                if target_fn is not None and arm.roll_stuck():
                    print(f"        [再試行] {arm.name}: wrist_roll が端で停止、手首を反転")
                    arm.flip_next = True
                    retry[key] = target_fn
            if retry:
                ok = self.run(Step(f"{step.name}（やり直し）", step.duration,
                                   feeder=retry.get("feeder"), builder=retry.get("builder"))) and ok
        return ok

    def print_status(self):
        # TCP 誤差・下向き度・干渉・可動域を表示
        errs = "  ".join(f"{k} {v.tcp_error() * 1000:5.1f} mm" for k, v in self.arms.items())
        downs = "  ".join(f"{k} {v.downward():5.2f}" for k, v in self.arms.items())
        print(f"        TCP 誤差  {errs}")
        print(f"        下向き度  {downs}")
        print(f"        干渉      {self.monitor.summary()}")
        for key, arm in self.arms.items():
            limits = arm.joints_at_limit()
            if limits:
                print(f"        可動域    {key}: {', '.join(limits)}")


def randomize_blocks(scene, origin: torch.Tensor, seed: int):
    # ブロックの初期姿勢をランダム化
    gen = torch.Generator().manual_seed(seed)
    for key, start in (("blue", T.BLUE_START), ("red", T.RED_START)):
        block = scene[key]
        r = torch.rand(3, generator=gen) * 2.0 - 1.0
        pos = torch.tensor([start], device=origin.device) + origin
        pos[0, 0] += r[0].item() * T.RANDOM_XY
        pos[0, 1] += r[1].item() * T.RANDOM_XY
        yaw = math.pi / 2 + math.radians(r[2].item() * T.RANDOM_YAW_DEG)
        quat = yaw_to_quat(yaw, origin.device)
        block.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        block.write_root_velocity_to_sim(torch.zeros(1, 6, device=origin.device))
        rel = (pos - origin)[0].tolist()
        print(f"[初期位置] {key}: ({rel[0]:.3f}, {rel[1]:.3f}) m、{math.degrees(yaw):.1f} deg")


def make_perceive(name: str, scene, block, detector, perceived: dict, runner) -> Callable[[], bool]:
    # ブロックの検出と姿勢推定の処理を作成
    camera = scene["d435i_color"]
    query = T.BLOCK_QUERIES[name]

    def perceive() -> bool:
        # 内部パラメータ未設定ならカメラを再初期化
        if camera.data.intrinsic_matrices[0, 0, 0].item() == 0.0:
            camera.reset()

        # 最新の画像を取得
        runner.settle(0.1)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        depth = camera.data.output["distance_to_image_plane"][0, ..., 0].clone()
        if args.depth_noise:
            # 距離比例の深度ノイズを付加
            depth = depth + torch.randn_like(depth) * depth * T.DEPTH_NOISE_REL
        # 最小測距距離未満の深度を除外
        depth[depth < T.D435I_DEPTH_MIN] = 0.0
        intrinsic = camera.data.intrinsic_matrices[0]
        cam_pos = camera.data.pos_w[0]
        cam_quat = camera.data.quat_w_ros[0]

        # 物体検出で枠を取得
        found = detector.detect(rgb, query)
        image_path = os.path.join(args.image_dir, f"detect_{query.replace(' ', '_')}.png")
        if found is None:
            save_detection_image(image_path, rgb, None, 0.0, query, None, 0.0, intrinsic, cam_pos, cam_quat, 0.0)
            print(f"        [認識失敗] {name}: 「{query}」未検出")
            return False
        box, score = found

        # 深度から位置と向きを推定
        est = estimate_block_pose(
            box, depth, intrinsic, cam_pos, cam_quat, T.TOP_Z_RANGE, T.BLOCK_HALF_HEIGHT,
            T.BLOCK_SIZE[0], T.OCCLUSION_TOL,
        )
        if est is None:
            save_detection_image(image_path, rgb, box, score, query, None, 0.0, intrinsic, cam_pos, cam_quat, 0.0)
            print(f"        [認識失敗] {name}: 上面の点が不足")
            return False
        center, yaw, n_points, occluded = est
        if occluded:
            print(f"        [認識] {name}: 片端遮蔽、見えている端から中心を推定")
        save_detection_image(
            image_path, rgb, box, score, query, center + torch.tensor([0.0, 0.0, T.BLOCK_HALF_HEIGHT], device=center.device),
            yaw, intrinsic, cam_pos, cam_quat, T.BLOCK_SIZE[0] / 2,
        )
        perceived[name] = (center.unsqueeze(0), yaw_to_quat(yaw, center.device))

        # 正解値との誤差を計算
        true_pos, true_quat = block_pose(block)
        true_axis = block_axis(true_quat)[0]
        true_yaw = math.atan2(true_axis[1].item(), true_axis[0].item())
        yaw_err = abs(math.remainder(yaw - true_yaw, math.pi))
        pos_err = (center - true_pos[0])[:2].norm().item()
        print(f"        [認識] {name}: スコア {score:.2f}、点 {n_points} 個、"
              f"位置誤差 {pos_err * 1000:.1f} mm、向き誤差 {math.degrees(yaw_err):.1f} deg")
        return True

    return perceive


def report_standing(scene, origin: torch.Tensor):
    # 立てた T を評価
    blue_pos, blue_quat = block_pose(scene["blue"])
    red_pos, red_quat = block_pose(scene["red"])
    stem = torch.tensor([T.STAND_STEM_CENTER], device=blue_pos.device) + origin
    bar = torch.tensor([T.STAND_BAR_CENTER], device=red_pos.device) + origin

    stem_err = (blue_pos - stem).norm().item()
    bar_err = (red_pos - bar).norm().item()
    upright = abs(block_axis(blue_quat)[0, 2].item())
    level = abs(block_axis(red_quat)[0, 2].item())
    offset = (red_pos - blue_pos)[0, :2].norm().item()

    print("========== 結果（立てた T） ==========")
    print(f"青の位置誤差   {stem_err * 1000:6.1f} mm")
    print(f"青の垂直度     {upright:6.3f}")
    print(f"赤の位置誤差   {bar_err * 1000:6.1f} mm")
    print(f"赤の高さ       {(red_pos - origin)[0, 2].item() * 1000:6.1f} mm（目標 {T.STAND_BAR_CENTER[2] * 1000:.1f} mm）")
    print(f"赤の傾き       {level:6.3f}")
    print(f"中心の水平ずれ {offset * 1000:6.1f} mm")
    ok = stem_err < 0.015 and bar_err < 0.015 and upright > 0.95 and level < 0.2
    print("判定: 成功" if ok else "判定: 失敗")


def report(scene, origin: torch.Tensor):
    # 平置きの T を評価
    blue_pos, blue_quat = block_pose(scene["blue"])
    red_pos, red_quat = block_pose(scene["red"])
    stem = torch.tensor([T.STEM_CENTER], device=blue_pos.device) + origin
    bar = torch.tensor([T.BAR_CENTER], device=red_pos.device) + origin

    stem_err = (blue_pos - stem).norm().item()
    bar_err = (red_pos - bar).norm().item()
    cross = (horizontal_unit(block_axis(blue_quat)) * horizontal_unit(block_axis(red_quat))).sum().abs().item()

    print("========== 結果 ==========")
    print(f"青の位置誤差 {stem_err * 1000:6.1f} mm")
    print(f"赤の位置誤差 {bar_err * 1000:6.1f} mm")
    print(f"赤の高さ     {(red_pos - origin)[0, 2].item() * 1000:6.1f} mm（目標 {T.BAR_CENTER[2] * 1000:.1f} mm）")
    print(f"直交度       {cross:6.3f}")
    ok = stem_err < 0.015 and bar_err < 0.015 and cross < 0.2
    print("判定: 成功" if ok else "判定: 失敗")


def main():
    # シミュレーションを作成
    sim_cfg = sim_utils.SimulationCfg(
        dt=T.PHYSICS_DT,
        device=args.device,
        physx=PhysxCfg(gpu_max_rigid_patch_count=2**18),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.6, -0.5, 0.45], target=[0.0, 0.0, 0.05])

    # シーンを作成
    scene_cfg = T.TTaskSceneCfg(num_envs=1, env_spacing=2.0)
    if args.perception or args.side_camera:
        # D435i カメラを追加
        scene_cfg.d435i_color = T.make_d435i_camera_cfg(depth=args.perception)
    if args.actuator != "ideal":
        # アクチュエータ設定を置換
        scene_cfg.feeder.actuators = make_actuators(args.actuator)
        scene_cfg.builder.actuators = make_actuators(args.actuator)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # シーンをリセット
    scene.reset()
    scene.update(sim.get_physics_dt())
    origin = scene.env_origins[0:1].clone()
    dev = origin.device

    # ブロックの初期姿勢をランダム化
    if args.randomize:
        randomize_blocks(scene, origin, args.seed)

    # 腕の制御と干渉検出を作成
    feeder = ArmController("feeder", scene["feeder"], torch.tensor([T.FEEDER_BASE], device=dev) + origin,
                           T.FEEDER_REST_POSE)
    builder = ArmController("builder", scene["builder"], torch.tensor([T.BUILDER_BASE], device=dev) + origin,
                            T.REST_POSE)
    arms = {"feeder": feeder, "builder": builder}
    if args.actuator == "real":
        # IK を運動学モデル上で解く設定
        for arm in arms.values():
            arm.ik.i_limit = T.REAL_IK_I_LIMIT
            arm.ik.i_gain = T.REAL_IK_I_GAIN
            arm.ik.i_decay = T.REAL_IK_I_DECAY
            diff = arm.ik.enable_model(SO101_URDF_PATH)
            print(f"[モデル] {arm.name}: 手先位置の差 {diff * 1000:.2f} mm")
    blocks = {"青": scene["blue"], "赤": scene["red"]}
    block_sensors = {"feeder": scene["feeder_block_contact"], "builder": scene["builder_block_contact"]}
    monitor = ContactMonitor(scene["arm_contact"], scene["builder"], block_sensors, arms, blocks)
    runner = Runner(sim, scene, arms, blocks, monitor)

    # 休息姿勢を書き込んで静止
    for arm in arms.values():
        joint_pos = arm.robot.data.default_joint_pos.clone()
        for i, name in enumerate(arm.robot.joint_names):
            joint_pos[:, i] = arm.rest_pose[name]
        arm.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    runner.settle(1.0, use_ik=False)

    # 休息姿勢を IK 指令に設定
    for arm in arms.values():
        arm.setup_grasp_center()
        arm.capture()

    # D435i 画像の保存を開始
    if args.side_camera:
        runner.recorder = SideRecorder(scene["d435i_color"], args.side_dir, args.side_fps, runner.dt)
        runner.recorder.active = True
        print(f"[D435i] 保存先 {args.side_dir}/（{args.side_fps:.0f} 枚/秒）")

    # 休息姿勢の確認のみ
    if args.check_ready:
        if runner.recorder is not None:
            # 確認用に数枚保存
            runner.settle(0.1)
            runner.recorder.active = False
            print(f"[D435i] {runner.recorder.count} 枚保存: {args.side_dir}/")
        while simulation_app.is_running():
            runner.settle(0.1)
        return

    # 認識を準備
    perceived: dict[str, Pose] = {}
    perceive_fns = {name: None for name in blocks}
    if args.perception:
        print("[認識] 物体検出モデルを読み込み中")
        detector = BlockDetector(device=str(dev), threshold=args.det_threshold)
        os.makedirs(args.image_dir, exist_ok=True)
        for name in blocks:
            perceive_fns[name] = make_perceive(name, scene, blocks[name], detector, perceived, runner)

    def pick_pose_fn(name: str) -> Callable[[], Pose]:
        return lambda: perceived[name] if name in perceived else block_pose(blocks[name])

    # 手順を作成
    handover_center = torch.tensor([T.HANDOVER_CENTER], device=dev) + origin
    stem_center = torch.tensor([T.STEM_CENTER], device=dev) + origin
    bar_center = torch.tensor([T.BAR_CENTER], device=dev) + origin
    stem_dir = torch.tensor([T.STEM_DIR], device=dev)
    bar_dir = torch.tensor([T.BAR_DIR], device=dev)
    steps = []
    if args.shape == "standing":
        # 青を立てて赤を載せる
        low_center = torch.tensor([T.STAND_HANDOVER_CENTER], device=dev) + origin
        stand_stem = torch.tensor([T.STAND_STEM_CENTER], device=dev) + origin
        stand_bar = torch.tensor([T.STAND_BAR_CENTER], device=dev) + origin
        stand_bar_dir = torch.tensor([T.STAND_BAR_DIR], device=dev)
        steps += make_standing_stem_steps(feeder, builder, scene["blue"], "青", low_center, stand_stem,
                                          pick_pose_fn("青"), perceive_fns["青"])
        steps += make_block_steps(feeder, builder, scene["red"], "赤", handover_center, stand_bar, stand_bar_dir,
                                  pick_pose_fn("赤"), perceive_fns["赤"],
                                  place_hover=T.BAR_ON_STAND_HOVER, retreat=T.STAND_RETREAT)
    else:
        # 平置きの T を作成
        steps += make_block_steps(feeder, builder, scene["blue"], "青", handover_center, stem_center, stem_dir,
                                  pick_pose_fn("青"), perceive_fns["青"])
        steps += make_block_steps(feeder, builder, scene["red"], "赤", handover_center, bar_center, bar_dir,
                                  pick_pose_fn("赤"), perceive_fns["赤"])

    # 手順を実行
    for step in steps:
        ok = runner.run(step)
        if not simulation_app.is_running():
            return
        if ok is False and not args.continue_on_fail:
            print("[停止] 把持失敗")
            while simulation_app.is_running():
                runner.settle(0.1)
            return

    # 両腕を休息姿勢へ移動
    runner.run(Step("両腕が休息姿勢へ", 2.5, home=("feeder", "builder"), home_to="rest"))
    runner.settle(1.0)
    print(f"[時間] {runner.sim_time:.1f} 秒（--speed {args.speed}）")
    if runner.recorder is not None:
        runner.recorder.active = False
        print(f"[D435i] {runner.recorder.count} 枚保存: {args.side_dir}/")
    if args.shape == "standing":
        report_standing(scene, origin)
    else:
        report(scene, origin)

    # ウィンドウを閉じるまで待機
    if args.hold:
        while simulation_app.is_running():
            runner.settle(0.1)


if __name__ == "__main__":
    main()
    simulation_app.close()
