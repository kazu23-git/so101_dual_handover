from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    subtract_frame_transforms,
)

from .arm_kinematics import BatchArm
from .handover_env_cfg import HandoverEnvCfg

GRIPPER = 5


class HandoverEnv(DirectRLEnv):
    cfg: HandoverEnvCfg

    def __init__(self, cfg: HandoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        dev = self.device

        # 関節とリンクのインデックスを取得
        self._giver_joint_ids, _ = self.giver.find_joints(cfg.joint_names, preserve_order=True)
        self._recv_joint_ids, _ = self.receiver.find_joints(cfg.joint_names, preserve_order=True)
        self._giver_hand_id = self.giver.find_bodies(cfg.hand_body_name)[0][0]
        self._recv_hand_id = self.receiver.find_bodies(cfg.hand_body_name)[0][0]

        # 関節の可動域を取得
        recv_limits = self.receiver.data.soft_joint_pos_limits[:, self._recv_joint_ids]
        self._recv_lower = recv_limits[..., 0]
        self._recv_upper = recv_limits[..., 1]
        giver_limits = self.giver.data.soft_joint_pos_limits[:, self._giver_joint_ids]
        self._giver_lower = giver_limits[..., 0]
        self._giver_upper = giver_limits[..., 1]

        # 初期姿勢をテンソルに変換
        self._giver_nominal = torch.tensor(
            [cfg.giver_present_pos[name] for name in cfg.joint_names], device=dev
        )
        self._giver_noise = torch.tensor(cfg.giver_present_noise, device=dev)
        self._recv_nominal = torch.tensor(
            [cfg.receiver_init_pos[name] for name in cfg.joint_names], device=dev
        )

        # 関節目標と行動のバッファを確保
        self._recv_targets = torch.zeros(n, 6, device=dev)
        self._giver_targets = torch.zeros(n, 6, device=dev)
        self._actions = torch.zeros(n, cfg.action_space, device=dev)
        self._prev_actions = torch.zeros(n, cfg.action_space, device=dev)

        # 受け渡しの状態を確保
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)
        self._just_released = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasp_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._squeezing = torch.zeros(n, dtype=torch.bool, device=dev)
        self._holding = torch.zeros(n, dtype=torch.bool, device=dev)
        self._success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=dev)

        # 受け取る側での固定位置を確保
        self._recv_attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._block_in_recv_pos = torch.zeros(n, 3, device=dev)
        self._block_in_recv_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)

        # エピソード結果のバッファを確保
        self._outcomes = torch.zeros(cfg.outcome_window, 5, device=dev)
        self._outcome_next = 0
        self._outcome_count = 0

        # 座標変換の定数を作成
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        self._tcp_offset = torch.tensor(cfg.tcp_offset, device=dev).repeat(n, 1)
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(n, 1)

        # 両側の把持点を計算
        half_span = cfg.grip_offset
        self._block_in_tcp_quat = torch.tensor(cfg.block_in_tcp_quat, device=dev).repeat(n, 1)
        giver_grip_local = torch.tensor([half_span, 0.0, 0.0], device=dev).repeat(n, 1)
        self._block_in_tcp_pos = -quat_apply(self._block_in_tcp_quat, giver_grip_local)
        self._top = cfg.grasp_mode == "top"
        grasp_x = -(cfg.block_size[0] / 2 - cfg.top_grasp_depth) if self._top else -half_span
        self._grasp_local = torch.tensor([grasp_x, 0.0, 0.0], device=dev).repeat(n, 1)
        self._y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(n, 1)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)

        # 爪の間の範囲を設定
        self._jaw_window = torch.tensor(cfg.grasp_window, device=dev)

        # 引き寄せ位置を設定
        self._carry_target = torch.tensor(cfg.carry_target, device=dev).repeat(n, 1)

        # 参照軌道のバッファを確保
        self._arm = BatchArm(cfg.urdf_path, cfg.hand_body_name, list(cfg.joint_names[:GRIPPER]), cfg.tcp_offset, dev)
        self._pre_nominal = torch.tensor(cfg.pre_pose, device=dev)
        self._recv_nominal_pose = torch.tensor(cfg.recv_pose, device=dev)
        self._carry_nominal = torch.tensor(cfg.carry_pose, device=dev)
        self._q_pre = self._pre_nominal.repeat(n, 1)
        self._q_recv = self._recv_nominal_pose.repeat(n, 1)
        self._q_carry = self._carry_nominal.repeat(n, 1)
        self._ref = torch.zeros(n, GRIPPER, device=dev)
        self._ref_start = torch.zeros(n, GRIPPER, device=dev)
        self._ref_from = torch.zeros(n, GRIPPER, device=dev)
        self._release_time = torch.zeros(n, device=dev)
        self._phase = torch.zeros(n, 2, device=dev)
        self._up = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(n, 1)

        # 中間値を計算
        self._update_frames()

    # ------------------------------------------------------------------
    # シーン
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # ロボットとブロックを作成
        self.giver = Articulation(self.cfg.giver_cfg)
        self.receiver = Articulation(self.cfg.receiver_cfg)
        self.block = RigidObject(self.cfg.block_cfg)

        # 床と照明を追加
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        # 環境を複製
        self.scene.clone_environments(copy_from_source=False)

        # シーンに登録
        self.scene.articulations["giver"] = self.giver
        self.scene.articulations["receiver"] = self.receiver
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
            self._recv_targets[:, :GRIPPER] = self._ref + self._actions * self.cfg.residual_limit
        else:
            self._recv_targets[:, :GRIPPER] += self._actions * self.cfg.action_scale
        self._recv_targets[:] = torch.clamp(self._recv_targets, self._recv_lower, self._recv_upper)

        # 受け取る側のグリッパ目標を更新
        close = self._in_jaws | self._released
        self._recv_targets[:, GRIPPER] = torch.where(
            close,
            torch.full_like(self._recv_targets[:, GRIPPER], self.cfg.gripper_close_target),
            torch.full_like(self._recv_targets[:, GRIPPER], self.cfg.gripper_open_angle),
        )

        # 渡す側のグリッパ目標を更新
        self._giver_targets[:, GRIPPER] = torch.where(
            self._released,
            torch.full_like(self._giver_targets[:, GRIPPER], self.cfg.giver_gripper_open),
            torch.full_like(self._giver_targets[:, GRIPPER], self.cfg.giver_gripper_closed),
        )

    def _update_reference(self):
        # ブロックの位置と向きから受け取る姿勢を更新
        cfg = self.cfg
        root_pos = self.receiver.data.root_pos_w
        root_quat = self.receiver.data.root_quat_w
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._grasp_pos)
        axis_b = quat_apply(quat_conjugate(root_quat), self._block_axis)
        flat = grasp_b.clone()
        flat[:, 2] = 0.0
        back = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        carry_b, _ = subtract_frame_transforms(root_pos, root_quat, self._carry_target + self.scene.env_origins)
        q_pre, q_recv, q_carry = self._q_pre, self._q_recv, self._q_carry
        if self._top:
            # 上端を真上からつまむ（手先を長手方向に、爪の幅を側面に合わせる）
            pre_b = grasp_b - back * cfg.pre_distance + self._up * cfg.top_pre_up
            carry_b = carry_b + self._up * cfg.top_grasp_depth
            side_b = quat_apply(quat_conjugate(root_quat), self._jaw_side)
            w = cfg.top_rot_weight
            for _ in range(cfg.ik_iters):
                q_pre = self._arm.step(q_pre, pre_b, side_b, -axis_b, rot_weight=w)
                q_recv = self._arm.step(q_recv, grasp_b, side_b, -axis_b, rot_weight=w)
                q_carry = self._arm.step(q_carry, carry_b, None, self._up, rot_weight=w)
        else:
            pre_b = grasp_b - back * cfg.pre_distance
            carry_b = carry_b + self._up * cfg.grip_offset
            for _ in range(cfg.ik_iters):
                q_pre = self._arm.step(q_pre, pre_b, axis_b)
                q_recv = self._arm.step(q_recv, grasp_b, axis_b)
                q_carry = self._arm.step(q_carry, carry_b, self._up)
        lower = self._recv_lower[:, :GRIPPER]
        upper = self._recv_upper[:, :GRIPPER]
        keep = self._released.unsqueeze(-1)
        self._q_pre = torch.where(keep, self._q_pre, torch.clamp(q_pre, lower, upper))
        self._q_recv = torch.where(keep, self._q_recv, torch.clamp(q_recv, lower, upper))
        self._q_carry = torch.clamp(q_carry, lower, upper)

        # 経過時間から参照軌道を計算
        t = self.episode_length_buf.float() * self.step_dt
        a = _smoothstep(t / cfg.ref_pre_time).unsqueeze(-1)
        b = _smoothstep((t - cfg.ref_pre_time) / cfg.ref_recv_time).unsqueeze(-1)
        before = self._ref_start + (self._q_pre - self._ref_start) * a + (self._q_recv - self._q_pre) * b
        s = _smoothstep((t - self._release_time) / cfg.ref_carry_time).unsqueeze(-1)
        after = self._ref_from + (self._q_carry - self._ref_from) * s
        self._ref = torch.where(keep, after, before)

        # 参照の進捗を計算
        self._phase[:, 0] = (t / (cfg.ref_pre_time + cfg.ref_recv_time)).clamp(0.0, 1.0)
        self._phase[:, 1] = torch.where(self._released, s.squeeze(-1), torch.zeros_like(t))

    def _apply_action(self):
        # 関節目標を送信
        self.receiver.set_joint_position_target(self._recv_targets, joint_ids=self._recv_joint_ids)
        self.giver.set_joint_position_target(self._giver_targets, joint_ids=self._giver_joint_ids)

        # 受け渡し前のブロックを渡す側に固定
        held_ids = (~self._released).nonzero(as_tuple=False).squeeze(-1)
        if held_ids.numel() > 0:
            self._attach_block(self.giver, self._giver_hand_id, held_ids,
                             self._block_in_tcp_pos[held_ids], self._block_in_tcp_quat[held_ids])

        # 受け渡し後のブロックを受け取る側に固定
        recv_ids = (self._released & self._recv_attached).nonzero(as_tuple=False).squeeze(-1)
        if recv_ids.numel() > 0:
            self._attach_block(self.receiver, self._recv_hand_id, recv_ids,
                             self._block_in_recv_pos[recv_ids], self._block_in_recv_quat[recv_ids])

    def _attach_block(self, robot: Articulation, hand_id: int, env_ids: torch.Tensor,
                    rel_pos: torch.Tensor, rel_quat: torch.Tensor):
        # TCP の姿勢を計算
        hand_pos = robot.data.body_pos_w[env_ids, hand_id]
        hand_quat = robot.data.body_quat_w[env_ids, hand_id]
        tcp_pos, tcp_quat = combine_frame_transforms(
            hand_pos, hand_quat, self._tcp_offset[env_ids], self._identity_quat[env_ids]
        )

        # ブロックの姿勢を計算
        block_pos, block_quat = combine_frame_transforms(tcp_pos, tcp_quat, rel_pos, rel_quat)

        # ブロックの姿勢を更新
        self.block.write_root_pose_to_sim(torch.cat([block_pos, block_quat], dim=-1), env_ids=env_ids)

    # ------------------------------------------------------------------
    # 座標の計算
    # ------------------------------------------------------------------

    def _record_recv_grasp(self, env_ids: torch.Tensor):
        # 受け取る側での固定位置を設定
        if self._top:
            # 手先の向きに長手方向を合わせて固定
            rel_pos = torch.zeros(len(env_ids), 3, device=self.device)
            rel_pos[:, 2] = -self.cfg.top_grasp_depth
            rel_quat = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=self.device).repeat(len(env_ids), 1)
            self._block_in_recv_pos[env_ids] = rel_pos
            self._block_in_recv_quat[env_ids] = rel_quat
            return
        sign = torch.where(self._align_signed[env_ids] >= 0, 1.0, -1.0)
        rel_pos = torch.zeros(len(env_ids), 3, device=self.device)
        rel_pos[:, 0] = self.cfg.grasp_center_x
        rel_pos[:, 1] = sign * self.cfg.grip_offset
        half = 0.7071068
        rel_quat = torch.zeros(len(env_ids), 4, device=self.device)
        rel_quat[:, 0] = half
        rel_quat[:, 3] = sign * half
        self._block_in_recv_pos[env_ids] = rel_pos
        self._block_in_recv_quat[env_ids] = rel_quat

    def _tcp_pose(self, robot: Articulation, hand_id: int):
        # TCP の位置と姿勢を計算
        hand_pos = robot.data.body_pos_w[:, hand_id]
        hand_quat = robot.data.body_quat_w[:, hand_id]
        return combine_frame_transforms(hand_pos, hand_quat, self._tcp_offset, self._identity_quat)

    def _update_frames(self):
        # 両腕の TCP を計算
        self._giver_tcp_pos, _ = self._tcp_pose(self.giver, self._giver_hand_id)
        self._recv_tcp_pos, recv_tcp_quat = self._tcp_pose(self.receiver, self._recv_hand_id)
        self._recv_tcp_y = quat_apply(recv_tcp_quat, self._y_axis)
        self._recv_approach = -quat_apply(recv_tcp_quat, self._z_axis)

        # ブロックの把持点と長手方向を計算
        block_pos = self.block.data.root_pos_w
        block_quat = self.block.data.root_quat_w
        self._block_pos = block_pos
        self._grasp_pos = block_pos + quat_apply(block_quat, self._grasp_local)
        self._block_axis = quat_apply(block_quat, self._x_axis)

        # TCP と把持点の距離を計算
        self._dist = torch.norm(self._recv_tcp_pos - self._grasp_pos, dim=-1)

        # 爪とブロックの向きの一致度を計算
        self._align_signed = torch.sum(self._recv_tcp_y * self._block_axis, dim=-1)
        self._align = torch.abs(self._align_signed)
        if self._top:
            # 手先が長手方向を向き、爪の幅が側面にそろっているか
            by = quat_apply(block_quat, self._y_axis)
            bz = quat_apply(block_quat, self._z_axis)
            cy = torch.sum(self._recv_tcp_y * by, dim=-1)
            cz = torch.sum(self._recv_tcp_y * bz, dim=-1)
            use_y = (cy.abs() >= cz.abs()).unsqueeze(-1)
            self._jaw_side = torch.where(use_y, by, bz)
            self._facing = torch.sum(self._recv_approach * self._block_axis, dim=-1)
            self._cross = torch.maximum(cy.abs(), cz.abs())
            self._align = self._facing.clamp(min=0.0) * self._cross
            self._align_signed = self._align

        # 爪の間の判定を実施
        self._grasp_in_tcp = quat_apply(quat_conjugate(recv_tcp_quat), self._grasp_pos - self._recv_tcp_pos)
        self._inside = torch.all(torch.abs(self._grasp_in_tcp) < self._jaw_window, dim=-1)
        self._aligned = self._align > self.cfg.grasp_align_cos
        if self._top:
            self._aligned = (self._facing > self.cfg.grasp_align_cos) & (self._cross > self.cfg.grasp_align_cos)
        self._in_jaws = self._inside & self._aligned

        # 引き寄せ位置までの距離を計算
        carry_w = self._carry_target + self.scene.env_origins
        self._carry_dist = torch.norm(self._block_pos - carry_w, dim=-1)

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_frames()

        # 受け取る側の土台の座標系に変換
        root_pos = self.receiver.data.root_pos_w
        root_quat = self.receiver.data.root_quat_w
        tcp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._recv_tcp_pos)
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._grasp_pos)
        axis_b = quat_apply(quat_conjugate(root_quat), self._block_axis)
        tcp_dir = self._recv_approach if self._top else self._recv_tcp_y
        tcp_y_b = quat_apply(quat_conjugate(root_quat), tcp_dir)

        # 関節の状態を取得
        joint_pos = self.receiver.data.joint_pos[:, self._recv_joint_ids]
        joint_vel = self.receiver.data.joint_vel[:, self._recv_joint_ids]
        joint_pos_rel = joint_pos - self._recv_nominal

        # 観測を結合（35 次元）
        obs = torch.cat(
            [
                joint_pos_rel,
                joint_vel,
                self._actions,
                tcp_b,
                grasp_b,
                grasp_b - tcp_b,
                axis_b,
                tcp_y_b,
                self._released.float().unsqueeze(-1),
                self._phase,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # 終了判定と受け渡しの状態更新
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_frames()
        cfg = self.cfg

        # 把持を判定
        gripper_pos = self.receiver.data.joint_pos[:, self._recv_joint_ids[GRIPPER]]
        gripper_vel = self.receiver.data.joint_vel[:, self._recv_joint_ids[GRIPPER]]
        gripper_target = self._recv_targets[:, GRIPPER]
        self._closed = gripper_pos < cfg.gripper_closed_max
        self._stalled = torch.abs(gripper_vel) < cfg.gripper_stall_vel
        self._pushing = gripper_target < gripper_pos - cfg.squeeze_margin
        self._squeezing = self._in_jaws & self._closed

        # 把持の継続ステップ数を更新
        counting = self._squeezing & ~self._released
        self._grasp_steps = torch.where(counting, self._grasp_steps + 1, torch.zeros_like(self._grasp_steps))

        # 受け渡しを判定
        self._just_released = ~self._released & (self._grasp_steps >= cfg.release_steps)
        self._released |= self._just_released
        just_ids = self._just_released.nonzero(as_tuple=False).squeeze(-1)
        if just_ids.numel() > 0:
            self._record_recv_grasp(just_ids)
            self._recv_attached[just_ids] = True
            # 参照軌道を引き寄せに切替
            self._ref_from[just_ids] = self._ref[just_ids]
            self._release_time[just_ids] = self.episode_length_buf[just_ids].float() * self.step_dt

        # 保持を判定
        self._holding = self._released & self._recv_attached
        carried = self._holding & (self._carry_dist < cfg.carry_tol)
        self._hold_steps = torch.where(carried, self._hold_steps + 1, torch.zeros_like(self._hold_steps))

        # 成功と落下を判定
        self._success = self._hold_steps == cfg.success_hold_steps
        self._succeeded |= self._success
        self._dropped = self._released & ~self._recv_attached

        terminated = self._dropped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # 報酬
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        # 把持点への接近
        r_reach = 1.0 - torch.tanh(self._dist / cfg.reach_std)

        # 爪の向きの一致
        r_align = r_reach * self._align

        # 爪の間への進入
        r_in_jaws = (self._in_jaws | self._released).float()

        # 引き寄せ位置への運搬
        r_carry = self._holding.float() * (1.0 - torch.tanh(self._carry_dist / cfg.carry_std))

        # 把持、受け渡し、保持、成功、落下
        r_grasp = self._squeezing.float()
        r_release = self._just_released.float()
        r_hold = self._holding.float()
        r_success = self._success.float()
        r_drop = self._dropped.float()

        # 両手の接近のペナルティ
        hand_gap = torch.norm(self._recv_tcp_pos - self._giver_tcp_pos, dim=-1)
        r_clearance = (hand_gap < cfg.min_hand_clearance).float()

        # 動きの滑らかさ
        r_action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        joint_vel = self.receiver.data.joint_vel[:, self._recv_joint_ids]
        r_joint_vel = torch.sum(joint_vel**2, dim=-1)

        reward = (
            cfg.rew_reach * r_reach
            + cfg.rew_align * r_align
            + cfg.rew_in_jaws * r_in_jaws
            + cfg.rew_grasp * r_grasp
            + cfg.rew_release * r_release
            + cfg.rew_hold * r_hold
            + cfg.rew_carry * r_carry
            + cfg.rew_success * r_success
            + cfg.rew_drop * r_drop
            + cfg.rew_clearance * r_clearance
            + cfg.rew_action_rate * r_action_rate
            + cfg.rew_joint_vel * r_joint_vel
        )

        # 学習の経過を記録
        count = max(1, min(self._outcome_count, self.cfg.outcome_window))
        rates = self._outcomes[:count].mean(dim=0).tolist()
        self.extras["log"] = {
            "reach_dist": self._dist.mean().item(),
            "released_rate": self._released.float().mean().item(),
            "success_rate": self._success.float().mean().item(),
            "drop_rate": self._dropped.float().mean().item(),
            "ep_success": rates[0],
            "ep_released": rates[1],
            "ep_drop": rates[2],
            "ep_drop_before_release": rates[3],
            "ep_length": rates[4],
        }
        return reward

    def _record_outcomes(self, env_ids: torch.Tensor):
        # エピソードの結果を記録
        ended = env_ids[self.episode_length_buf[env_ids] > 0]
        if ended.numel() == 0:
            return
        rows = torch.stack(
            [
                self._succeeded[ended].float(),
                self._released[ended].float(),
                self._dropped[ended].float(),
                (self._dropped[ended] & ~self._released[ended]).float(),
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
            env_ids = self.receiver._ALL_INDICES
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # 受け取る側の初期姿勢を設定
        noise = (torch.rand(n, 6, device=self.device) * 2.0 - 1.0) * self.cfg.receiver_init_noise
        recv_q = self._recv_nominal.unsqueeze(0) + noise
        recv_q = torch.clamp(recv_q, self._recv_lower[env_ids], self._recv_upper[env_ids])
        self.receiver.write_joint_state_to_sim(
            recv_q, torch.zeros_like(recv_q), joint_ids=self._recv_joint_ids, env_ids=env_ids
        )
        self._recv_targets[env_ids] = recv_q

        # 参照軌道を初期化
        self._ref_start[env_ids] = recv_q[:, :GRIPPER]
        self._ref[env_ids] = recv_q[:, :GRIPPER]
        self._ref_from[env_ids] = recv_q[:, :GRIPPER]
        self._q_pre[env_ids] = self._pre_nominal
        self._q_recv[env_ids] = self._recv_nominal_pose
        self._q_carry[env_ids] = self._carry_nominal
        self._release_time[env_ids] = 0.0
        self._phase[env_ids] = 0.0

        # 渡す側の提示姿勢をサンプリング
        noise = (torch.rand(n, 6, device=self.device) * 2.0 - 1.0) * self._giver_noise
        giver_q = self._giver_nominal.unsqueeze(0) + noise
        giver_q[:, GRIPPER] = self.cfg.giver_gripper_closed
        giver_q = torch.clamp(giver_q, self._giver_lower[env_ids], self._giver_upper[env_ids])
        self.giver.write_joint_state_to_sim(
            giver_q, torch.zeros_like(giver_q), joint_ids=self._giver_joint_ids, env_ids=env_ids
        )
        self._giver_targets[env_ids] = giver_q

        # ブロックの位置をリセット
        block_state = self.block.data.default_root_state[env_ids].clone()
        block_state[:, :3] += self.scene.env_origins[env_ids]
        self.block.write_root_pose_to_sim(block_state[:, :7], env_ids=env_ids)

        # 受け渡しの状態と行動をリセット
        self._released[env_ids] = False
        self._just_released[env_ids] = False
        self._grasp_steps[env_ids] = 0
        self._hold_steps[env_ids] = 0
        self._succeeded[env_ids] = False
        self._recv_attached[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    # 3 次の補間係数を計算
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
