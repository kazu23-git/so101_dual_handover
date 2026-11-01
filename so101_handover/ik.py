"""DLS による IK と把持姿勢の計算。"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME, HAND_BODY_NAME

# TCP の位置（gripper_link 座標系）
TCP_OFFSET = (0.0, 0.0, -0.098)

# たわみ補正の対象関節
GRAVITY_JOINT_NAMES = ["shoulder_lift", "elbow_flex", "wrist_flex"]


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    # URDF の rpy を回転行列に変換
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    # 軸と角度を回転行列に変換
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    # 回転行列を回転ベクトルに変換
    cos = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos)
    vee = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    if angle < 1e-6:
        return vee * 0.5
    return vee * (angle / (2.0 * math.sin(angle)))


class KinematicChain:
    # URDF から運動学モデルを構築

    def __init__(self, urdf_path: str, end_link: str, joint_names: list[str]):
        root = ET.parse(urdf_path).getroot()
        by_child = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            ax = [float(v) for v in (axis.get("xyz", "1 0 0") if axis is not None else "1 0 0").split()]
            by_child[joint.find("child").get("link")] = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "xyz": np.array(xyz),
                "rot": _rpy_to_matrix(*rpy),
                "axis": np.array(ax) / max(np.linalg.norm(ax), 1e-9),
            }

        # 手先から土台まで探索
        chain = []
        link = end_link
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link]["parent"]
        chain.reverse()
        self.chain = chain
        self.root_link = link
        self.q_index = [
            joint_names.index(j["name"]) if j["type"] in ("revolute", "continuous") else -1 for j in chain
        ]

    def fk(self, q: np.ndarray) -> np.ndarray:
        # 土台座標系での手先姿勢を計算
        tf = np.eye(4)
        for joint, idx in zip(self.chain, self.q_index):
            step = np.eye(4)
            step[:3, :3] = joint["rot"]
            step[:3, 3] = joint["xyz"]
            tf = tf @ step
            if idx >= 0:
                rot = np.eye(4)
                rot[:3, :3] = _axis_angle_to_matrix(joint["axis"], q[idx])
                tf = tf @ rot
        return tf

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # 差分でヤコビアンを計算
        base = self.fk(q)
        jac = np.zeros((6, len(q)))
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += eps
            moved = self.fk(dq)
            jac[:3, i] = (moved[:3, 3] - base[:3, 3]) / eps
            jac[3:, i] = _matrix_to_rotvec(moved[:3, :3] @ base[:3, :3].T) / eps
        return jac


def _quat_to_matrix(quat: torch.Tensor) -> np.ndarray:
    # クォータニオン (w, x, y, z) を回転行列に変換
    w, x, y, z = quat.detach().cpu().double().tolist()
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class DlsIk:
    def __init__(
        self,
        robot: Articulation,
        damping: float = 0.05,
        max_step: float = 0.5,
        rot_weight: float = 0.3,
        i_gain: float = 0.02,
        i_limit: float = 0.2,
        i_decay: float = 0.998,
    ):
        self.robot = robot
        self.damping = damping
        self.max_step = max_step
        self.rot_weight = rot_weight
        self.i_gain = i_gain
        self.i_limit = i_limit
        self.i_decay = i_decay
        dev = robot.device

        # 関節とリンクのインデックスを取得
        self.arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.grip_id = robot.find_joints([GRIPPER_JOINT_NAME])[0][0]
        self.hand_id = robot.find_bodies(HAND_BODY_NAME)[0][0]
        # ヤコビアン用のリンク番号を設定
        self.jac_id = self.hand_id - 1

        self.tcp_offset = torch.tensor(TCP_OFFSET, device=dev).unsqueeze(0)

        # 積算項を確保
        self.dq_sum = torch.zeros(robot.num_instances, len(self.arm_ids), device=dev)
        self.i_mask = torch.tensor(
            [[1.0 if name in GRAVITY_JOINT_NAMES else 0.0 for name in ARM_JOINT_NAMES]], device=dev
        )

        # 運動学モデル用の状態を確保
        self.model: KinematicChain | None = None
        self.q_model: np.ndarray | None = None
        self.model_iters = 2
        self.last_des: torch.Tensor | None = None
        self.i_gate_step = 1e-4  # 積算する目標移動量の上限 [m]

    def set_tcp_offset(self, offset: tuple[float, float, float]):
        # TCP の位置を変更
        self.tcp_offset = torch.tensor(offset, device=self.tcp_offset.device).unsqueeze(0)

    def _offset(self, quat: torch.Tensor) -> torch.Tensor:
        # オフセットを台数分に拡張
        return self.tcp_offset.expand(quat.shape[0], 3)

    def reset_integral(self):
        # 積算値とモデル状態をリセット
        self.dq_sum.zero_()
        self.q_model = None
        self.last_des = None

    def enable_model(self, urdf_path: str) -> float:
        # 運動学モデルを有効化し、手先位置の差 [m] を返却
        self.model = KinematicChain(urdf_path, HAND_BODY_NAME, ARM_JOINT_NAMES)
        q = self.robot.data.joint_pos[0, self.arm_ids].detach().cpu().double().numpy()
        hand_model = self._root_to_world(self.model.fk(q))[0]
        hand_sim = self.robot.data.body_pos_w[0, self.hand_id].detach().cpu().double().numpy()
        return float(np.linalg.norm(hand_model - hand_sim))

    def _root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # 土台の位置と回転行列を取得
        pos = self.robot.data.root_pos_w[0].detach().cpu().double().numpy()
        return pos, _quat_to_matrix(self.robot.data.root_quat_w[0])

    def _root_to_world(self, tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 土台座標系の姿勢をワールド座標系に変換
        root_pos, root_rot = self._root_pose()
        return root_pos + root_rot @ tf[:3, 3], root_rot @ tf[:3, :3]

    def tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        # TCP の位置と姿勢を取得
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        return hand_pos + quat_apply(hand_quat, self._offset(hand_quat)), hand_quat

    def solve(self, tcp_pos_des: torch.Tensor, tcp_quat_des: torch.Tensor) -> torch.Tensor:
        # モデル有効時はモデル上で計算
        if self.model is not None:
            return self._solve_model(tcp_pos_des, tcp_quat_des)

        # TCP 目標を gripper_link 目標に変換
        hand_pos_des = tcp_pos_des - quat_apply(tcp_quat_des, self._offset(tcp_quat_des))

        # 姿勢誤差を計算
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        pos_err, rot_err = compute_pose_error(
            hand_pos, hand_quat, hand_pos_des, tcp_quat_des, rot_error_type="axis_angle"
        )
        err = torch.cat([pos_err, rot_err * self.rot_weight], dim=-1).unsqueeze(-1)

        # ヤコビアンを取得
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_id][:, :, self.arm_ids].clone()
        jac[:, 3:, :] *= self.rot_weight

        # 関節の変化量を計算
        jjt = jac @ jac.transpose(1, 2)
        eye = torch.eye(6, device=jac.device).unsqueeze(0)
        dq = jac.transpose(1, 2) @ torch.linalg.solve(jjt + (self.damping**2) * eye, err)
        dq = dq.squeeze(-1).clamp(-self.max_step, self.max_step)

        # 残留誤差を積算補正
        self.dq_sum = self.dq_sum * self.i_decay + self.i_gain * dq * self.i_mask
        self.dq_sum = self.dq_sum.clamp(-self.i_limit, self.i_limit)

        return self.robot.data.joint_pos[:, self.arm_ids] + dq + self.dq_sum

    def _solve_model(self, tcp_pos_des: torch.Tensor, tcp_quat_des: torch.Tensor) -> torch.Tensor:
        # モデル上で IK を計算
        if self.q_model is None:
            self.q_model = self.robot.data.joint_pos[0, self.arm_ids].detach().cpu().double().numpy()
        limits = self.robot.data.joint_pos_limits[0, self.arm_ids].detach().cpu().double().numpy()

        # 目標を土台座標系に変換
        root_pos, root_rot = self._root_pose()
        rot_des_w = _quat_to_matrix(tcp_quat_des[0])
        offset = self.tcp_offset[0].detach().cpu().double().numpy()
        hand_pos_des_w = tcp_pos_des[0].detach().cpu().double().numpy() - rot_des_w @ offset
        pos_des = root_rot.T @ (hand_pos_des_w - root_pos)
        rot_des = root_rot.T @ rot_des_w

        # モデル上で DLS を反復
        weight = np.array([1.0, 1.0, 1.0, self.rot_weight, self.rot_weight, self.rot_weight])
        q = self.q_model
        for _ in range(self.model_iters):
            tf = self.model.fk(q)
            err = np.concatenate([pos_des - tf[:3, 3], _matrix_to_rotvec(rot_des @ tf[:3, :3].T)]) * weight
            jac = self.model.jacobian(q) * weight[:, None]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + (self.damping**2) * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -self.max_step, self.max_step), limits[:, 0], limits[:, 1])
        self.q_model = q

        # 目標静止時のみたわみを積算補正
        moving = self.last_des is not None and (tcp_pos_des - self.last_des).norm().item() > self.i_gate_step
        self.last_des = tcp_pos_des.clone()
        tcp_meas, _ = self.tcp_pose()
        err_meas = (tcp_pos_des - tcp_meas).unsqueeze(-1)
        jac_pos = self.robot.root_physx_view.get_jacobians()[:, self.jac_id][:, :3, self.arm_ids]
        eye = torch.eye(3, device=jac_pos.device).unsqueeze(0)
        dq_meas = jac_pos.transpose(1, 2) @ torch.linalg.solve(
            jac_pos @ jac_pos.transpose(1, 2) + (self.damping**2) * eye, err_meas
        )
        gain = 0.0 if moving else self.i_gain
        self.dq_sum = self.dq_sum * self.i_decay + gain * dq_meas.squeeze(-1) * self.i_mask
        self.dq_sum = self.dq_sum.clamp(-self.i_limit, self.i_limit)

        q_target = torch.tensor(q, dtype=torch.float32, device=self.dq_sum.device).unsqueeze(0)
        return q_target + self.dq_sum


def horizontal_unit(vec: torch.Tensor) -> torch.Tensor:
    # 水平成分を正規化
    out = vec.clone()
    out[..., 2] = 0.0
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def top_down_quat(axis: torch.Tensor) -> torch.Tensor:
    # 真下向きの把持姿勢を計算
    y = horizontal_unit(axis)
    z = torch.zeros_like(y)
    z[..., 2] = 1.0
    x = torch.linalg.cross(y, z, dim=-1)
    rot = torch.stack([x, y, z], dim=-1)
    return quat_from_matrix(rot)


def top_down_quat_near(axis: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    # 現在姿勢に近い側の真下向き姿勢を計算
    y_now = quat_apply(current_quat, torch.tensor([[0.0, 1.0, 0.0]], device=axis.device))
    sign = torch.sign((horizontal_unit(y_now) * horizontal_unit(axis)).sum(dim=-1, keepdim=True))
    sign[sign == 0] = 1.0
    return top_down_quat(axis * sign)


def tcp_axis(quat: torch.Tensor, local: tuple[float, float, float]) -> torch.Tensor:
    # TCP 座標系の軸をワールド座標系で取得
    vec = torch.tensor([local], device=quat.device).expand(quat.shape[0], 3)
    return quat_apply(quat, vec)


def approach_dir(quat: torch.Tensor) -> torch.Tensor:
    # 手先の向きを取得
    return tcp_axis(quat, (0.0, 0.0, -1.0))


def horizontal_grasp_quat(approach: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    # 水平把持の姿勢を計算
    a = horizontal_unit(approach)
    up = torch.zeros_like(a)
    up[..., 2] = 1.0
    x = torch.linalg.cross(a, up, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    z = -a
    y = torch.linalg.cross(z, x, dim=-1)
    rot = torch.stack([x, y, z], dim=-1)
    return quat_from_matrix(rot)


def pitch_to_approach(quat: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    # X 軸回りに ±90 度回した姿勢から desired に近い方を選択
    x_axis = tcp_axis(quat, (1.0, 0.0, 0.0))
    best_quat, best_score = quat, -2.0
    for angle in (math.pi / 2, -math.pi / 2):
        angle_t = torch.full((quat.shape[0],), angle, device=quat.device)
        q = quat_mul(quat_from_angle_axis(angle_t, x_axis), quat)
        score = (approach_dir(q) * desired).sum().item()
        if score > best_score:
            best_quat, best_score = q, score
    return best_quat
