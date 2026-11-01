from __future__ import annotations

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    subtract_frame_transforms,
)

from .arm_kinematics import BatchArm
from .handover_env_cfg import RECEIVER_INIT_POS, HandoverEnvCfg
from .pick_env import GRIPPER, PickEnv, _smoothstep
from .pick_env_cfg import FEEDER_INIT_POS, PickEnvCfg
from .robot_cfg import make_so101_cfg
from .t_task_cfg import BASE_DISTANCE

_RECV = HandoverEnvCfg()


@configclass
class RelayEnvCfg(PickEnvCfg):
    # 両方の方策の行動を受け取る
    action_space = 10
    episode_length_s = 16.0

    # 組立側を受け取る側の初期姿勢で配置
    builder_cfg: ArticulationCfg = make_so101_cfg(
        prim_path="/World/envs/env_.*/Builder",
        pos=(0.0, -BASE_DISTANCE / 2, 0.0),
        yaw_deg=90.0,
        init_joint_pos=RECEIVER_INIT_POS,
        actuator="real",
    )

    # 掴んだときの姿勢のまま固定（台にめり込まないように）
    keep_grasp_pose = True

    # 受け取りを始める条件を設定
    start_present_steps = 10
    start_timeout = 3.0          # [s]
    feeder_return_time = 1.5     # [s]


class RelayEnv(PickEnv):
    # 供給側が掴んで差し出し、組立側が受け取る通しの環境
    cfg: RelayEnvCfg

    def __init__(self, cfg: RelayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        dev = self.device
        r = _RECV

        # 供給側の行動バッファを 5 次元に設定
        self._actions = torch.zeros(n, GRIPPER, device=dev)
        self._prev_actions = torch.zeros(n, GRIPPER, device=dev)

        # 組立側のリンクと可動域を取得
        self._recv_hand_id = self.builder.find_bodies(cfg.hand_body_name)[0][0]
        limits = self.builder.data.soft_joint_pos_limits[:, self._builder_joint_ids]
        self._recv_lower = limits[..., 0]
        self._recv_upper = limits[..., 1]
        self._recv_nominal = torch.tensor([RECEIVER_INIT_POS[k] for k in cfg.joint_names], device=dev)
        self._feeder_rest = torch.tensor([FEEDER_INIT_POS[k] for k in cfg.joint_names[:GRIPPER]], device=dev)

        # 受け取る側の参照軌道を準備
        self._arm_r = BatchArm(cfg.urdf_path, cfg.hand_body_name, list(cfg.joint_names[:GRIPPER]), r.tcp_offset, dev)
        self._pre_nominal = torch.tensor(r.pre_pose, device=dev)
        self._recv_pose_nominal = torch.tensor(r.recv_pose, device=dev)
        self._carry_nominal = torch.tensor(r.carry_pose, device=dev)
        self._q_pre = self._pre_nominal.repeat(n, 1)
        self._q_recv = self._recv_pose_nominal.repeat(n, 1)
        self._q_carry = self._carry_nominal.repeat(n, 1)
        self._r_ref = torch.zeros(n, GRIPPER, device=dev)
        self._r_ref_start = torch.zeros(n, GRIPPER, device=dev)
        self._r_ref_from = torch.zeros(n, GRIPPER, device=dev)
        self._r_phase = torch.zeros(n, 2, device=dev)
        self._r_targets = self._recv_nominal.repeat(n, 1)
        self._r_actions = torch.zeros(n, GRIPPER, device=dev)
        self._r_tcp_offset = torch.tensor(r.tcp_offset, device=dev).repeat(n, 1)
        self._carry_target = torch.tensor(r.carry_target, device=dev).repeat(n, 1)
        self._r_window = torch.tensor(r.grasp_window, device=dev)

        # 受け渡しの状態を確保
        self._active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._start_time = torch.zeros(n, device=dev)
        self._handed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hand_time = torch.zeros(n, device=dev)
        self._r_grasp_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._r_hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._r_in_jaws = torch.zeros(n, dtype=torch.bool, device=dev)
        self.relay_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._feeder_from = torch.zeros(n, GRIPPER, device=dev)
        self._block_in_recv_pos = torch.zeros(n, 3, device=dev)
        self._block_in_recv_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._update_recv_frames()

    # ------------------------------------------------------------------
    # 受け取る側の座標
    # ------------------------------------------------------------------

    def _update_recv_frames(self):
        r = _RECV
        hand_pos = self.builder.data.body_pos_w[:, self._recv_hand_id]
        hand_quat = self.builder.data.body_quat_w[:, self._recv_hand_id]
        self._r_tcp_pos, r_tcp_quat = combine_frame_transforms(hand_pos, hand_quat, self._r_tcp_offset, self._identity_quat)
        self._r_tcp_y = quat_apply(r_tcp_quat, self._y_axis)

        # 供給側が掴んでいない方の端を把持点に設定
        block_pos = self.block.data.root_pos_w
        axis = quat_apply(self.block.data.root_quat_w, self._x_axis)
        self._r_axis = axis * self._grip_side.unsqueeze(-1)
        self._r_grasp = block_pos - self._r_axis * r.grip_offset
        self._r_align_signed = torch.sum(self._r_tcp_y * self._r_axis, dim=-1)
        grasp_in_tcp = quat_apply(quat_conjugate(r_tcp_quat), self._r_grasp - self._r_tcp_pos)
        inside = torch.all(torch.abs(grasp_in_tcp) < self._r_window, dim=-1)
        self._r_in_jaws = self._active & ~self._handed & inside & (self._r_align_signed.abs() > r.grasp_align_cos)
        self._r_carry_dist = torch.norm(block_pos - (self._carry_target + self.scene.env_origins), dim=-1)

    def _update_recv_reference(self):
        r = _RECV
        root_pos = self.builder.data.root_pos_w
        root_quat = self.builder.data.root_quat_w
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._r_grasp)
        axis_b = quat_apply(quat_conjugate(root_quat), self._r_axis)
        flat = grasp_b.clone()
        flat[:, 2] = 0.0
        pre_b = grasp_b - flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6) * r.pre_distance
        carry_b, _ = subtract_frame_transforms(root_pos, root_quat, self._carry_target + self.scene.env_origins)
        carry_b = carry_b + self._z_axis * r.grip_offset
        q_pre, q_recv, q_carry = self._q_pre, self._q_recv, self._q_carry
        for _ in range(r.ik_iters):
            q_pre = self._arm_r.step(q_pre, pre_b, axis_b)
            q_recv = self._arm_r.step(q_recv, grasp_b, axis_b)
            q_carry = self._arm_r.step(q_carry, carry_b, self._z_axis)
        lower = self._recv_lower[:, :GRIPPER]
        upper = self._recv_upper[:, :GRIPPER]
        keep = self._handed.unsqueeze(-1)
        # 受け取りを始めるまでは初期値のまま（台の上や移動中のブロックを追わない）
        frozen = (self._handed | ~self._active).unsqueeze(-1)
        self._q_pre = torch.where(frozen, self._q_pre, torch.clamp(q_pre, lower, upper))
        self._q_recv = torch.where(frozen, self._q_recv, torch.clamp(q_recv, lower, upper))
        self._q_carry = torch.where(self._active.unsqueeze(-1), torch.clamp(q_carry, lower, upper), self._q_carry)

        # 受け取りを始めてからの時間で参照軌道を計算
        now = self.episode_length_buf.float() * self.step_dt
        t = now - self._start_time
        a = _smoothstep(t / r.ref_pre_time).unsqueeze(-1)
        b = _smoothstep((t - r.ref_pre_time) / r.ref_recv_time).unsqueeze(-1)
        before = self._r_ref_start + (self._q_pre - self._r_ref_start) * a + (self._q_recv - self._q_pre) * b
        s = _smoothstep((now - self._hand_time) / r.ref_carry_time).unsqueeze(-1)
        after = self._r_ref_from + (self._q_carry - self._r_ref_from) * s
        ref = torch.where(keep, after, before)
        self._r_ref = torch.where(self._active.unsqueeze(-1), ref, self._r_ref_start)
        self._r_phase[:, 0] = torch.where(self._active, (t / (r.ref_pre_time + r.ref_recv_time)).clamp(0.0, 1.0), 0.0)
        self._r_phase[:, 1] = torch.where(self._handed, s.squeeze(-1), torch.zeros_like(t))

    # ------------------------------------------------------------------
    # 行動
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        cfg = self.cfg
        r = _RECV

        # 供給側の関節目標を更新
        super()._pre_physics_step(actions[:, :GRIPPER])

        # 渡したあとの供給側を休息姿勢へ戻す
        now = self.episode_length_buf.float() * self.step_dt
        s = _smoothstep((now - self._hand_time) / cfg.feeder_return_time).unsqueeze(-1)
        back = self._feeder_from + (self._feeder_rest - self._feeder_from) * s
        self._targets[:, :GRIPPER] = torch.where(self._handed.unsqueeze(-1), back, self._targets[:, :GRIPPER])
        self._targets[:, GRIPPER] = torch.where(self._handed, torch.full_like(now, cfg.gripper_open_angle), self._targets[:, GRIPPER])

        # 受け取る側の関節目標を更新
        self._r_actions[:] = actions[:, GRIPPER:].clamp(-1.0, 1.0)
        self._update_recv_reference()
        residual = self._r_actions * r.residual_limit * self._active.float().unsqueeze(-1)
        self._r_targets[:, :GRIPPER] = torch.clamp(self._r_ref + residual, self._recv_lower[:, :GRIPPER], self._recv_upper[:, :GRIPPER])
        close = self._r_in_jaws | self._handed
        self._r_targets[:, GRIPPER] = torch.where(
            close, torch.full_like(now, r.gripper_close_target), torch.full_like(now, r.gripper_open_angle)
        )

    def _apply_action(self):
        # 関節目標を送信
        self.feeder.set_joint_position_target(self._targets, joint_ids=self._joint_ids)
        self.builder.set_joint_position_target(self._r_targets, joint_ids=self._builder_joint_ids)

        # 供給側が持っているブロックを固定
        ids = (self._attached & ~self._handed).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            tcp_pos, tcp_quat = self._tcp_pose(ids)
            pos, quat = combine_frame_transforms(tcp_pos, tcp_quat, self._block_in_tcp_pos[ids], self._block_in_tcp_quat[ids])
            self.block.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)

        # 受け取ったブロックを固定
        ids = self._handed.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            hand_pos = self.builder.data.body_pos_w[ids, self._recv_hand_id]
            hand_quat = self.builder.data.body_quat_w[ids, self._recv_hand_id]
            tcp_pos, tcp_quat = combine_frame_transforms(hand_pos, hand_quat, self._r_tcp_offset[ids], self._identity_quat[ids])
            pos, quat = combine_frame_transforms(tcp_pos, tcp_quat, self._block_in_recv_pos[ids], self._block_in_recv_quat[ids])
            self.block.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)

    # ------------------------------------------------------------------
    # 状態の更新
    # ------------------------------------------------------------------

    def _get_dones(self):
        cfg = self.cfg
        r = _RECV
        _, time_out = super()._get_dones()
        self._attached &= ~self._handed
        now = self.episode_length_buf.float() * self.step_dt

        # 差し出したら受け取りを開始
        ready = self._attached & ~self._active & (
            (self._present_steps >= cfg.start_present_steps) | (now - self._grasp_time > cfg.start_timeout)
        )
        ids = ready.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._active[ids] = True
            self._start_time[ids] = now[ids]
            self._r_ref_start[ids] = self.builder.data.joint_pos[ids][:, self._builder_joint_ids[:GRIPPER]]

        # 受け渡しを判定
        self._update_recv_frames()
        grip = self.builder.data.joint_pos[:, self._builder_joint_ids[GRIPPER]]
        squeezing = self._r_in_jaws & (grip < r.gripper_closed_max)
        self._r_grasp_steps = torch.where(squeezing, self._r_grasp_steps + 1, torch.zeros_like(self._r_grasp_steps))
        just = ~self._handed & (self._r_grasp_steps >= r.release_steps)
        ids = just.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            sign = torch.where(self._r_align_signed[ids] >= 0, 1.0, -1.0)
            rel_pos = torch.zeros(len(ids), 3, device=self.device)
            rel_pos[:, 0] = r.grasp_center_x
            rel_pos[:, 1] = sign * r.grip_offset
            rel_quat = torch.zeros(len(ids), 4, device=self.device)
            rel_quat[:, 0] = 0.7071068
            rel_quat[:, 3] = sign * 0.7071068
            self._block_in_recv_pos[ids] = rel_pos
            self._block_in_recv_quat[ids] = rel_quat
            self._handed[ids] = True
            self._hand_time[ids] = now[ids]
            self._r_ref_from[ids] = self._r_ref[ids]
            self._feeder_from[ids] = self.feeder.data.joint_pos[ids][:, self._joint_ids[:GRIPPER]]
            self._attached[ids] = False

        # 引き寄せて保ったら成功
        carried = self._handed & (self._r_carry_dist < r.carry_tol)
        self._r_hold_steps = torch.where(carried, self._r_hold_steps + 1, torch.zeros_like(self._r_hold_steps))
        self.relay_success |= self._r_hold_steps >= r.success_hold_steps
        return torch.zeros_like(time_out), time_out

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        self._update_recv_frames()
        root_pos = self.builder.data.root_pos_w
        root_quat = self.builder.data.root_quat_w
        inv = quat_conjugate(root_quat)
        tcp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._r_tcp_pos)
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._r_grasp)
        joint_pos = self.builder.data.joint_pos[:, self._builder_joint_ids]
        joint_vel = self.builder.data.joint_vel[:, self._builder_joint_ids]
        recv = torch.cat(
            [
                joint_pos - self._recv_nominal,
                joint_vel,
                self._r_actions,
                tcp_b,
                grasp_b,
                grasp_b - tcp_b,
                quat_apply(inv, self._r_axis),
                quat_apply(inv, self._r_tcp_y),
                self._handed.float().unsqueeze(-1),
                self._r_phase,
            ],
            dim=-1,
        )
        obs["receiver"] = recv
        return obs

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.feeder._ALL_INDICES
        super()._reset_idx(env_ids)
        rest = self._recv_nominal.unsqueeze(0).repeat(len(env_ids), 1)
        self._r_targets[env_ids] = rest
        self._r_ref_start[env_ids] = rest[:, :GRIPPER]
        self._r_ref[env_ids] = rest[:, :GRIPPER]
        self._r_ref_from[env_ids] = rest[:, :GRIPPER]
        self._q_pre[env_ids] = self._pre_nominal
        self._q_recv[env_ids] = self._recv_pose_nominal
        self._q_carry[env_ids] = self._carry_nominal
        self._r_phase[env_ids] = 0.0
        self._r_actions[env_ids] = 0.0
        self._active[env_ids] = False
        self._handed[env_ids] = False
        self._start_time[env_ids] = 0.0
        self._hand_time[env_ids] = 0.0
        self._r_grasp_steps[env_ids] = 0
        self._r_hold_steps[env_ids] = 0
        self.relay_success[env_ids] = False
