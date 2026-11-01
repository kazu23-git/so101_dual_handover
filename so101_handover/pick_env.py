from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from .gripper_points import child_points, link_points
from .ik import KinematicChain
from .pick_env_cfg import PickEnvCfg

GRIPPER = 5


class PickEnv(DirectRLEnv):
    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        dev = self.device

        # 関節とリンクのインデックスを取得
        self._joint_ids, _ = self.feeder.find_joints(cfg.joint_names, preserve_order=True)
        self._builder_joint_ids, _ = self.builder.find_joints(cfg.joint_names, preserve_order=True)
        self._hand_id = self.feeder.find_bodies(cfg.hand_body_name)[0][0]
        self._jaw_id = self.feeder.find_bodies(cfg.jaw_body_name)[0][0]

        # 爪の表面点を読み込み
        self._hand_points = torch.tensor(
            link_points(cfg.urdf_path, cfg.hand_body_name, cfg.penetration_voxel), dtype=torch.float32, device=dev
        )
        self._jaw_points = torch.tensor(
            link_points(cfg.urdf_path, cfg.jaw_body_name, cfg.penetration_voxel), dtype=torch.float32, device=dev
        )
        self._closed_jaw_points = torch.tensor(
            child_points(cfg.urdf_path, cfg.gripper_joint_name, cfg.gripper_close_target, cfg.penetration_voxel),
            dtype=torch.float32, device=dev,
        )
        self._block_inner = torch.tensor(cfg.block_size, device=dev) * 0.5 - cfg.penetration_margin

        # 関節の可動域を取得
        limits = self.feeder.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lower = limits[..., 0]
        self._upper = limits[..., 1]

        # 初期姿勢をテンソルに変換
        self._nominal = torch.tensor([cfg.feeder_init_pos[name] for name in cfg.joint_names], device=dev)
        self._builder_rest = self.builder.data.default_joint_pos[:, self._builder_joint_ids].clone()

        # 関節目標と行動のバッファを確保
        self._targets = torch.zeros(n, 6, device=dev)
        self._actions = torch.zeros(n, cfg.action_space, device=dev)
        self._prev_actions = torch.zeros(n, cfg.action_space, device=dev)

        # 掴む・差し出すの状態を確保
        self._attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._just_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._squeeze_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._squeezing = torch.zeros(n, dtype=torch.bool, device=dev)
        self._present_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._presenting = torch.zeros(n, dtype=torch.bool, device=dev)
        self._success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out_of_area = torch.zeros(n, dtype=torch.bool, device=dev)
        self._assisted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pen_hand = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pen_jaw = torch.zeros(n, dtype=torch.bool, device=dev)
        self._penetrating = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fits = torch.zeros(n, dtype=torch.bool, device=dev)
        self._snap_dist = torch.zeros(n, device=dev)

        # 掴む端と手先に対するブロックの相対姿勢を確保
        self._grip_side = torch.ones(n, device=dev)
        self._block_in_tcp_pos = torch.zeros(n, 3, device=dev)
        self._block_in_tcp_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)

        # 座標変換の定数を作成
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._tcp_offset = torch.tensor(cfg.tcp_offset, device=dev).repeat(n, 1)
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(n, 1)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        self._jaw_window = torch.tensor(cfg.grasp_window, device=dev)
        self._present_center = torch.tensor(cfg.present_center, device=dev).repeat(n, 1)
        self._present_approach = torch.tensor(cfg.present_approach, device=dev).repeat(n, 1)
        self._block_starts = torch.tensor(cfg.block_starts, device=dev)
        self._hover_poses = torch.tensor(cfg.hover_poses, device=dev)
        self._present_pose = torch.tensor(cfg.present_pose, device=dev)
        self._area_min = torch.tensor(cfg.area_min, device=dev)
        self._area_max = torch.tensor(cfg.area_max, device=dev)
        self._grasp_poses = torch.tensor(cfg.grasp_poses, device=dev)

        # 参照軌道と補正のバッファを確保
        self._choice = torch.zeros(n, dtype=torch.long, device=dev)
        self._ref = torch.zeros(n, GRIPPER, device=dev)
        self._ref_start = torch.zeros(n, GRIPPER, device=dev)
        self._ref_from = torch.zeros(n, GRIPPER, device=dev)
        self._grasp_time = torch.zeros(n, device=dev)
        self._t0 = torch.zeros(n, device=dev)
        self._residual = torch.zeros(n, GRIPPER, device=dev)
        self._phase = torch.zeros(n, 2, device=dev)
        self._ref_hover = self._hover_poses[self._choice].clone()
        self._ref_grasp = self._grasp_poses[self._choice].clone()

        # 参照姿勢を合わせる運動学モデルを作成
        self._chain = KinematicChain(cfg.urdf_path, cfg.hand_body_name, list(cfg.joint_names[:GRIPPER]))
        self._tcp_offset_np = np.array(cfg.tcp_offset)
        self._jinv_hover = [np.linalg.inv(self._jacobian(np.array(q))) for q in cfg.hover_poses]
        self._jinv_grasp = [np.linalg.inv(self._jacobian(np.array(q))) for q in cfg.grasp_poses]

        # エピソード結果のバッファを確保（成功・把持・落下・長さ）
        self._outcomes = torch.zeros(cfg.outcome_window, 4, device=dev)
        self._outcome_next = 0
        self._outcome_count = 0

        # 範囲外イベントの記録を確保
        self.out_events = 0
        self.out_after_grasp: list[float] = []
        self.out_positions: list[list[float]] = []

        # 中間値のバッファを確保
        self._update_frames()

    # ------------------------------------------------------------------
    # シーン
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # ロボットとブロックを作成
        self.feeder = Articulation(self.cfg.feeder_cfg)
        self.builder = Articulation(self.cfg.builder_cfg)
        self.block = RigidObject(self.cfg.block_cfg)

        # 床と照明を追加
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # 環境を複製
        self.scene.clone_environments(copy_from_source=False)

        # シーンに登録
        self.scene.articulations["feeder"] = self.feeder
        self.scene.articulations["builder"] = self.builder
        self.scene.rigid_objects["block"] = self.block

    # ------------------------------------------------------------------
    # 行動
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        # 行動を保存
        self._prev_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)

        # 腕の関節目標を更新
        if self.cfg.use_reference:
            # 参照軌道に補正を加算
            self._update_reference()
            # 補正量を制限
            limit = torch.where(
                self._attached, torch.full_like(self._grasp_time, self.cfg.residual_limit_after),
                torch.full_like(self._grasp_time, self.cfg.residual_limit),
            ).unsqueeze(-1)
            if self.cfg.residual_mode == "direct":
                # 行動をそのまま補正に使用
                self._residual = self._actions * limit
            else:
                # 行動を積算して補正に使用
                self._residual = torch.maximum(torch.minimum(self._residual + self._actions * self.cfg.action_scale, limit), -limit)
            self._targets[:, :GRIPPER] = self._ref + self._residual
        else:
            # 行動を関節目標に加算
            self._targets[:, :GRIPPER] += self._actions * self.cfg.action_scale
        self._targets[:] = torch.clamp(self._targets, self._lower, self._upper)

        # グリッパの開閉を設定
        close = (self._in_jaws & self._fits) | self._attached
        self._targets[:, GRIPPER] = torch.where(
            close,
            torch.full_like(self._targets[:, GRIPPER], self.cfg.gripper_close_target),
            torch.full_like(self._targets[:, GRIPPER], self.cfg.gripper_open_angle),
        )

    def _update_reference(self):
        # 経過時間から参照軌道を計算
        cfg = self.cfg
        t = self.episode_length_buf.float() * self.step_dt - self._t0
        hover = self._ref_hover
        grasp = self._ref_grasp

        a = _smoothstep(t / cfg.ref_hover_time).unsqueeze(-1)
        b = _smoothstep((t - cfg.ref_hover_time) / cfg.ref_descend_time).unsqueeze(-1)
        before = self._ref_start + (hover - self._ref_start) * a
        before = before + (grasp - hover) * b

        if cfg.present_via_hover:
            # 真上へ持ち上げてから差し出す
            u = _smoothstep((t - self._grasp_time) / cfg.ref_lift_time).unsqueeze(-1)
            s = _smoothstep((t - self._grasp_time - cfg.ref_lift_time) / cfg.ref_present_time).unsqueeze(-1)
            after = self._ref_from + (hover - self._ref_from) * u + (self._present_pose.unsqueeze(0) - hover) * s
        else:
            s = _smoothstep((t - self._grasp_time) / cfg.ref_present_time).unsqueeze(-1)
            after = self._ref_from + (self._present_pose.unsqueeze(0) - self._ref_from) * s

        self._ref = torch.where(self._attached.unsqueeze(-1), after, before)

        # 参照の進捗を計算
        self._phase[:, 0] = ((t / (cfg.ref_hover_time + cfg.ref_descend_time)).clamp(0.0, 1.0))
        self._phase[:, 1] = torch.where(self._attached, s.squeeze(-1), torch.zeros_like(t))

    def _apply_action(self):
        # 関節目標を送信
        self.feeder.set_joint_position_target(self._targets, joint_ids=self._joint_ids)
        self.builder.set_joint_position_target(self._builder_rest, joint_ids=self._builder_joint_ids)

        # 掴んだブロックを手先に固定
        ids = self._attached.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            tcp_pos, tcp_quat = self._tcp_pose(ids)
            pos, quat = combine_frame_transforms(
                tcp_pos, tcp_quat, self._block_in_tcp_pos[ids], self._block_in_tcp_quat[ids]
            )
            self.block.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)

    # ------------------------------------------------------------------
    # 座標の計算
    # ------------------------------------------------------------------

    def _tcp_pose(self, env_ids: torch.Tensor | None = None):
        # TCP の位置と姿勢を計算
        if env_ids is None:
            hand_pos = self.feeder.data.body_pos_w[:, self._hand_id]
            hand_quat = self.feeder.data.body_quat_w[:, self._hand_id]
            return combine_frame_transforms(hand_pos, hand_quat, self._tcp_offset, self._identity_quat)
        hand_pos = self.feeder.data.body_pos_w[env_ids, self._hand_id]
        hand_quat = self.feeder.data.body_quat_w[env_ids, self._hand_id]
        return combine_frame_transforms(
            hand_pos, hand_quat, self._tcp_offset[env_ids], self._identity_quat[env_ids]
        )

    def _tcp_fk(self, q: np.ndarray):
        # TCP の位置と姿勢を順運動学で計算（土台座標系）
        tf = self._chain.fk(q)
        return tf[:3, :3] @ self._tcp_offset_np + tf[:3, 3], tf[:3, :3]

    @staticmethod
    def _expand(d: np.ndarray) -> np.ndarray:
        # 手先の傾きを保つ 5 関節の変化量に変換
        return np.array([d[0], d[1], d[2], -(d[1] + d[2]), 0.0])

    def _jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        # TCP 位置のヤコビアンを数値微分で計算
        p0, _ = self._tcp_fk(q)
        jac = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            jac[:, i] = (self._tcp_fk(q + self._expand(d))[0] - p0) / eps
        return jac

    def _fit_pose(self, q0: np.ndarray, jinv: np.ndarray, target: np.ndarray, yaw: float) -> np.ndarray:
        # TCP を目標位置へ、爪の幅の向きをブロックに合わせる
        q = q0.copy()
        for _ in range(2):
            p, _ = self._tcp_fk(q)
            q = q + self._expand(jinv @ (target - p))
            _, rot = self._tcp_fk(q)
            err = yaw - math.atan2(rot[1, 1], rot[0, 1])
            q[4] += (err + math.pi / 2) % math.pi - math.pi / 2
        return q

    def _fit_reference(self, env_ids: torch.Tensor, block_pos: torch.Tensor, block_axis: torch.Tensor):
        # ブロックの位置と向きから真上と掴む姿勢を計算
        cfg = self.cfg
        root_pos = self.feeder.data.root_pos_w[env_ids]
        root_quat = self.feeder.data.root_quat_w[env_ids]
        grasp_w = block_pos + block_axis * (self._grip_side[env_ids] * cfg.grip_offset).unsqueeze(-1)
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, grasp_w)
        axis_b = quat_apply(quat_conjugate(root_quat), block_axis)
        yaw_b = torch.atan2(axis_b[:, 1], axis_b[:, 0])
        targets = grasp_b.cpu().numpy()
        yaws = yaw_b.cpu().numpy()
        choices = self._choice[env_ids].cpu().numpy()
        hover = np.zeros((len(choices), GRIPPER))
        grasp = np.zeros((len(choices), GRIPPER))
        lift = np.array([0.0, 0.0, cfg.hover_height])
        for i, c in enumerate(choices):
            grasp[i] = self._fit_pose(np.array(cfg.grasp_poses[c]), self._jinv_grasp[c], targets[i], yaws[i])
            hover[i] = self._fit_pose(np.array(cfg.hover_poses[c]), self._jinv_hover[c], targets[i] + lift, yaws[i])
        self._ref_grasp[env_ids] = torch.tensor(grasp, dtype=torch.float32, device=self.device)
        self._ref_hover[env_ids] = torch.tensor(hover, dtype=torch.float32, device=self.device)

    def _points_in_block(self, body_id: int, points: torch.Tensor) -> torch.Tensor:
        # リンクの表面点がブロック内にあるかを判定
        block_rot = matrix_from_quat(self.block.data.root_quat_w)
        body_rot = matrix_from_quat(self.feeder.data.body_quat_w[:, body_id])
        rot = block_rot.transpose(1, 2) @ body_rot
        shift = (block_rot.transpose(1, 2) @ (self.feeder.data.body_pos_w[:, body_id] - self.block.data.root_pos_w).unsqueeze(-1)).squeeze(-1)
        local = torch.einsum("nij,pj->npi", rot, points) + shift.unsqueeze(1)
        return torch.all(torch.abs(local) < self._block_inner, dim=-1).any(dim=-1)

    def _update_penetration(self):
        # 掴む前の爪の食い込みを計算
        free = ~self._attached
        hand_in = self._points_in_block(self._hand_id, self._hand_points)
        self._pen_hand = free & hand_in
        self._pen_jaw = free & self._points_in_block(self._jaw_id, self._jaw_points)
        self._penetrating = self._pen_hand | self._pen_jaw
        # 閉じた爪がブロックに入らないかを判定
        self._fits = ~hand_in & ~self._points_in_block(self._hand_id, self._closed_jaw_points)

    def _block_pose(self):
        # ブロックの位置と姿勢を取得
        return self.block.data.root_pos_w, self.block.data.root_quat_w

    def _update_frames(self):
        cfg = self.cfg

        # TCP の位置と軸方向を計算
        self._tcp_pos, self._tcp_quat = self._tcp_pose()
        self._tcp_y = quat_apply(self._tcp_quat, self._y_axis)
        self._approach = -quat_apply(self._tcp_quat, self._z_axis)

        # ブロックの中心、長手方向、掴む位置を計算
        self._block_pos, block_quat = self._block_pose()
        self._block_axis = quat_apply(block_quat, self._x_axis)
        self._grasp_pos = self._block_pos + self._block_axis * (self._grip_side * cfg.grip_offset).unsqueeze(-1)
        self._dist = torch.norm(self._tcp_pos - self._grasp_pos, dim=-1)

        # 把持条件を計算
        self._align_signed = torch.sum(self._tcp_y * self._block_axis, dim=-1)
        self._align = torch.abs(self._align_signed)
        self._down = -self._approach[:, 2]
        self._grasp_in_tcp = quat_apply(quat_conjugate(self._tcp_quat), self._grasp_pos - self._tcp_pos)
        self._inside = torch.all(torch.abs(self._grasp_in_tcp) < self._jaw_window, dim=-1)
        self._in_jaws = (
            self._inside & (self._align > cfg.grasp_align_cos) & (self._down > cfg.grasp_down_cos)
        )

        # 差し出す姿勢との差を計算
        target = self._present_center + self.scene.env_origins
        self._present_dist = torch.norm(self._block_pos - target, dim=-1)
        self._vertical = torch.abs(self._block_axis[:, 2])
        self._facing = torch.sum(self._approach * self._present_approach, dim=-1)
        # 下端把持の度合いを計算
        self._lower_end = ((self._block_pos[:, 2] - self._tcp_pos[:, 2]) / cfg.grip_offset).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_frames()

        # 供給側の土台の座標系に変換
        root_pos = self.feeder.data.root_pos_w
        root_quat = self.feeder.data.root_quat_w
        inv = quat_conjugate(root_quat)
        tcp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._tcp_pos)
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._grasp_pos)
        center_b, _ = subtract_frame_transforms(root_pos, root_quat, self._block_pos)
        target_b, _ = subtract_frame_transforms(
            root_pos, root_quat, self._present_center + self.scene.env_origins
        )
        approach_b = quat_apply(inv, self._approach)
        tcp_y_b = quat_apply(inv, self._tcp_y)
        axis_b = quat_apply(inv, self._block_axis)

        # 関節の状態を取得
        joint_pos = self.feeder.data.joint_pos[:, self._joint_ids]
        joint_vel = self.feeder.data.joint_vel[:, self._joint_ids]

        # 観測を結合（49 次元）
        obs = torch.cat(
            [
                joint_pos - self._nominal,
                joint_vel,
                self._actions,
                tcp_b,
                approach_b,
                tcp_y_b,
                grasp_b,
                grasp_b - tcp_b,
                axis_b,
                center_b,
                target_b - center_b,
                self._attached.float().unsqueeze(-1),
                self._residual,
                self._phase,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # 終了判定と状態の更新
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_frames()
        cfg = self.cfg

        # 把持判定を計算
        grip_pos = self.feeder.data.joint_pos[:, self._joint_ids[GRIPPER]]
        grip_vel = self.feeder.data.joint_vel[:, self._joint_ids[GRIPPER]]
        grip_target = self._targets[:, GRIPPER]
        self._closed = grip_pos < cfg.gripper_closed_max
        self._stalled = torch.abs(grip_vel) < cfg.gripper_stall_vel
        self._pushing = grip_target < grip_pos - cfg.squeeze_margin
        self._update_penetration()
        self._squeezing = ~self._attached & self._in_jaws & self._closed
        if cfg.grasp_reject_penetration:
            self._squeezing &= ~self._penetrating
        self._squeeze_steps = torch.where(
            self._squeezing, self._squeeze_steps + 1, torch.zeros_like(self._squeeze_steps)
        )

        # 把持を確定
        self._just_grasped = ~self._attached & (self._squeeze_steps >= cfg.grasp_steps)
        ids = self._just_grasped.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._record_grasp(ids)
            self._attached[ids] = True
            self._grasped[ids] = True
            # 参照軌道を切替
            self._ref_from[ids] = self._ref[ids]
            self._grasp_time[ids] = self.episode_length_buf[ids].float() * self.step_dt - self._t0[ids]
            # 補正をリセット
            if self.cfg.reset_residual_on_grasp:
                self._residual[ids] = 0.0

        # 落下フラグを更新
        self._dropped = torch.zeros_like(self._attached)
        self._attached &= ~self._dropped

        # 差し出し判定を更新
        self._presenting = (
            self._attached
            & (self._present_dist < cfg.present_tol)
            & (self._vertical > cfg.present_vertical_cos)
            & (self._facing > cfg.present_approach_cos)
            & (self._lower_end > 0.5)
        )
        self._present_steps = torch.where(
            self._presenting, self._present_steps + 1, torch.zeros_like(self._present_steps)
        )
        self._success = self._present_steps == cfg.success_hold_steps
        self._succeeded |= self._success

        # 作業範囲外を判定
        local = self._block_pos - self.scene.env_origins
        outside = torch.any((local < self._area_min) | (local > self._area_max), dim=-1)
        self._out_of_area = self._attached & outside

        # 範囲外イベントを記録
        if self._out_of_area.any():
            out = self._out_of_area
            t = self.episode_length_buf[out].float() * self.step_dt
            self.out_events += int(out.sum().item())
            self.out_after_grasp.extend((t - self._grasp_time[out]).tolist())
            self.out_positions.extend(local[out].tolist())

        terminated = self._dropped | self._out_of_area
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _record_grasp(self, env_ids: torch.Tensor):
        # 手先に対するブロックの相対姿勢を記録
        if self.cfg.keep_grasp_pose:
            self._canon_pos, self._canon_quat = self._canonical_rel(env_ids)
            self._record_grasp_as_is(env_ids)
            return
        rel_pos, rel_quat = self._canonical_rel(env_ids)
        self._block_in_tcp_pos[env_ids] = rel_pos
        self._block_in_tcp_quat[env_ids] = rel_quat

        # 固定時のブロックの移動量を記録
        tcp_pos, tcp_quat = self._tcp_pose(env_ids)
        pos, _ = combine_frame_transforms(tcp_pos, tcp_quat, rel_pos, rel_quat)
        self._snap_dist[env_ids] = torch.norm(pos - self._block_pos[env_ids], dim=-1)

    def _canonical_rel(self, env_ids: torch.Tensor):
        # 爪の間の決まった相対姿勢を計算
        sign = torch.where(self._align_signed[env_ids] >= 0, 1.0, -1.0)
        side = self._grip_side[env_ids]
        rel_pos = torch.zeros(len(env_ids), 3, device=self.device)
        rel_pos[:, 0] = self.cfg.grasp_center_x
        rel_pos[:, 1] = -side * sign * self.cfg.grip_offset
        half = 0.7071068
        rel_quat = torch.zeros(len(env_ids), 4, device=self.device)
        rel_quat[:, 0] = half
        rel_quat[:, 3] = sign * half
        return rel_pos, rel_quat

    def _record_grasp_as_is(self, env_ids: torch.Tensor):
        # 掴んだときの姿勢のまま固定（爪の閉じる方向だけ中心へ寄せ、面を爪にそろえる）
        tcp_pos, tcp_quat = self._tcp_pose(env_ids)
        rel_pos, rel_quat = subtract_frame_transforms(
            tcp_pos, tcp_quat, self.block.data.root_pos_w[env_ids], self.block.data.root_quat_w[env_ids]
        )
        rel_pos[:, 0] = self.cfg.grasp_center_x
        axes = matrix_from_quat(rel_quat)
        k = axes[:, 0, :].abs().argmax(dim=-1)
        normal = axes[torch.arange(len(env_ids), device=self.device), :, k]
        normal = normal * torch.where(normal[:, :1] >= 0, 1.0, -1.0)
        x = torch.zeros_like(normal)
        x[:, 0] = 1.0
        rot_axis = torch.cross(normal, x, dim=-1)
        sin = rot_axis.norm(dim=-1)
        angle = torch.atan2(sin, torch.sum(normal * x, dim=-1))
        rot_axis = torch.where(sin.unsqueeze(-1) > 1e-6, rot_axis / sin.clamp(min=1e-6).unsqueeze(-1), x)
        rel_quat = quat_mul(quat_from_angle_axis(angle, rot_axis), rel_quat)
        self._block_in_tcp_pos[env_ids] = rel_pos
        self._block_in_tcp_quat[env_ids] = rel_quat
        pos, _ = combine_frame_transforms(tcp_pos, tcp_quat, rel_pos, rel_quat)
        self._snap_dist[env_ids] = torch.norm(pos - self.block.data.root_pos_w[env_ids], dim=-1)

    # ------------------------------------------------------------------
    # 報酬
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        before = (~self._attached).float()
        after = self._attached.float()

        # 掴む前の報酬を計算
        r_reach = before * (1.0 - torch.tanh(self._dist / cfg.reach_std))
        near_wide = before * (1.0 - torch.tanh(self._dist / cfg.orient_std))
        r_align = near_wide * 0.5 * (self._down.clamp(0.0, 1.0) + self._align)

        r_in_jaws = (self._in_jaws & ~self._attached).float()

        # 掴んだあとは満点に設定
        r_reach = r_reach + after
        r_align = r_align + after
        r_in_jaws = r_in_jaws + after

        # 把持報酬を計算
        r_grasp = self._just_grasped.float()

        # 掴んだあとの報酬を計算
        r_hold = after
        lift = ((self._block_pos[:, 2] - cfg.block_half_height) / cfg.lift_height).clamp(0.0, 1.0)
        r_lift = after * lift
        r_present_coarse = after * (1.0 - torch.tanh(self._present_dist / cfg.present_std_coarse))
        r_present_fine = after * (1.0 - torch.tanh(self._present_dist / cfg.present_std))
        r_present = r_present_coarse + r_present_fine
        r_pose = r_present_coarse * self._vertical * self._facing.clamp(0.0, 1.0) * self._lower_end

        # 成功と失敗の報酬を計算
        r_success = self._success.float()
        r_drop = self._dropped.float()
        r_out = self._out_of_area.float()
        r_penetration = self._penetrating.float()

        # 補正のペナルティを計算
        r_residual = torch.sum(self._residual**2, dim=-1)

        # 滑らかさのペナルティを計算
        r_action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        joint_vel = self.feeder.data.joint_vel[:, self._joint_ids]
        r_joint_vel = torch.sum(joint_vel**2, dim=-1)

        reward = (
            cfg.rew_reach * r_reach
            + cfg.rew_align * r_align
            + cfg.rew_in_jaws * r_in_jaws
            + cfg.rew_grasp * r_grasp
            + cfg.rew_hold * r_hold
            + cfg.rew_lift * r_lift
            + cfg.rew_present * r_present
            + cfg.rew_present_pose * r_pose
            + cfg.rew_success * r_success
            + cfg.rew_drop * r_drop
            + cfg.rew_out_of_area * r_out
            + cfg.rew_penetration * r_penetration
            + cfg.rew_residual * r_residual
            + cfg.rew_action_rate * r_action_rate
            + cfg.rew_joint_vel * r_joint_vel
        )

        # ログを更新
        count = max(1, min(self._outcome_count, cfg.outcome_window))
        rates = self._outcomes[:count].mean(dim=0).tolist()
        self.extras["log"] = {
            "grasp_dist": (self._dist * before).sum().item() / max(1.0, before.sum().item()),
            "present_dist": (self._present_dist * after).sum().item() / max(1.0, after.sum().item()),
            "attached_rate": after.mean().item(),
            "penetration_rate": (self._penetrating.float() * before).sum().item() / max(1.0, before.sum().item()),
            "ep_success": rates[0],
            "ep_grasped": rates[1],
            "ep_drop": rates[2],
            "ep_length": rates[3],
        }
        return reward

    def _record_outcomes(self, env_ids: torch.Tensor):
        # エピソード結果を記録
        ended = env_ids[(self.episode_length_buf[env_ids] > 0) & ~self._assisted[env_ids]]
        if ended.numel() == 0:
            return
        rows = torch.stack(
            [
                self._succeeded[ended].float(),
                self._grasped[ended].float(),
                self._dropped[ended].float(),
                self.episode_length_buf[ended].float(),
            ],
            dim=-1,
        )
        size = self.cfg.outcome_window
        idx = (torch.arange(rows.shape[0], device=self.device) + self._outcome_next) % size
        self._outcomes[idx[-size:]] = rows[-size:]
        self._outcome_next = (self._outcome_next + rows.shape[0]) % size
        self._outcome_count += rows.shape[0]

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.feeder._ALL_INDICES
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        cfg = self.cfg
        dev = self.device

        # 置き場所を選択
        choice = torch.randint(0, self._block_starts.shape[0], (n,), device=dev)

        # 供給側の初期姿勢を設定
        r = torch.rand(n, device=dev)
        assisted = r < cfg.attached_start_prob
        hover = ~assisted & (r < cfg.attached_start_prob + cfg.hover_start_prob)
        start = self._nominal.unsqueeze(0).repeat(n, 1)
        start[hover, :GRIPPER] = self._hover_poses[choice[hover]]
        t = torch.rand(n, 1, device=dev)
        between = self._hover_poses[choice] * (1.0 - t) + self._present_pose.unsqueeze(0) * t
        start[assisted, :GRIPPER] = between[assisted]
        start[assisted, GRIPPER] = cfg.gripper_hold_angle
        noise = (torch.rand(n, 6, device=dev) * 2.0 - 1.0) * cfg.feeder_init_noise
        q = torch.clamp(start + noise, self._lower[env_ids], self._upper[env_ids])
        self.feeder.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self._joint_ids, env_ids=env_ids)

        # 参照軌道を初期化
        self._choice[env_ids] = choice
        self._ref_start[env_ids] = q[:, :GRIPPER]
        self._ref[env_ids] = q[:, :GRIPPER]
        self._ref_from[env_ids] = q[:, :GRIPPER]
        self._grasp_time[env_ids] = 0.0
        self._t0[env_ids] = 0.0
        self._residual[env_ids] = 0.0
        self._phase[env_ids] = 0.0
        self._targets[env_ids] = q
        self._targets[env_ids, GRIPPER] = torch.where(
            assisted, torch.full_like(q[:, 0], cfg.gripper_close_target), torch.full_like(q[:, 0], cfg.gripper_open_angle)
        )

        # 組立側をリセット
        rest = self._builder_rest[env_ids]
        self.builder.write_joint_state_to_sim(
            rest, torch.zeros_like(rest), joint_ids=self._builder_joint_ids, env_ids=env_ids
        )

        # ブロックの初期姿勢を設定
        pos = self._block_starts[choice].clone()
        pos[:, :2] += (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * cfg.block_random_xy
        pos += self.scene.env_origins[env_ids]
        yaw = math.radians(cfg.block_start_yaw_deg) + (
            (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(cfg.block_random_yaw_deg)
        )
        quat = quat_from_angle_axis(yaw, self._z_axis[env_ids])
        self.block.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

        # 掴む端を選択
        axis = quat_apply(quat, self._x_axis[env_ids])
        base = self.feeder.data.root_pos_w[env_ids]
        plus = torch.norm((pos + axis * cfg.grip_offset - base)[:, :2], dim=-1)
        minus = torch.norm((pos - axis * cfg.grip_offset - base)[:, :2], dim=-1)
        self._grip_side[env_ids] = torch.where(plus < minus, 1.0, -1.0)

        # 把持開始の環境の相対姿勢を設定
        self._grip_side[env_ids] = torch.where(assisted, 1.0, self._grip_side[env_ids])
        rel_pos = torch.zeros(n, 3, device=dev)
        rel_pos[:, 0] = cfg.grasp_center_x
        rel_pos[:, 1] = -cfg.grip_offset
        rel_quat = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=dev).repeat(n, 1)
        self._block_in_tcp_pos[env_ids] = torch.where(assisted.unsqueeze(-1), rel_pos, self._block_in_tcp_pos[env_ids])
        self._block_in_tcp_quat[env_ids] = torch.where(assisted.unsqueeze(-1), rel_quat, self._block_in_tcp_quat[env_ids])

        # 参照姿勢をブロックに合わせる
        self._ref_hover[env_ids] = self._hover_poses[choice]
        self._ref_grasp[env_ids] = self._grasp_poses[choice]
        if cfg.adapt_reference:
            self._fit_reference(env_ids, pos, axis)

        # 状態と行動をリセット
        self._attached[env_ids] = assisted
        self._assisted[env_ids] = assisted
        self._out_of_area[env_ids] = False
        self._just_grasped[env_ids] = False
        self._squeeze_steps[env_ids] = 0
        self._present_steps[env_ids] = 0
        self._succeeded[env_ids] = False
        self._grasped[env_ids] = False
        self._dropped[env_ids] = False
        self._pen_hand[env_ids] = False
        self._pen_jaw[env_ids] = False
        self._penetrating[env_ids] = False
        self._fits[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # 3 次の補間係数を計算
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
