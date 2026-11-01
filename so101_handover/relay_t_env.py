from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    subtract_frame_transforms,
)

from .arm_kinematics import BatchArm
from .handover_env_cfg import RECEIVER_INIT_POS, HandoverEnvCfg, HandoverTopEnvCfg
from .pick_env import GRIPPER, PickEnv, _smoothstep
from .pick_env_cfg import FEEDER_INIT_POS, LOW_PRESENT_POSE, PRESENT_POSE, PickEnvCfg
from .robot_cfg import make_so101_cfg
from .t_task_cfg import (
    BASE_DISTANCE,
    D435I_MOUNT_POS,
    D435I_MOUNT_USD,
    BLOCK_MASS,
    BLOCK_SIZE,
    BLUE_START,
    HANDOVER_CENTER,
    RANDOM_XY,
    RANDOM_YAW_DEG,
    RED_START,
    STAND_BAR_CENTER,
    STAND_HANDOVER_CENTER,
    STAND_STEM_CENTER,
    make_d435i_camera_cfg,
)

# 置く姿勢の初期値を定義 [rad]（真上と置く位置）
STEM_HOVER_POSE = (0.811, -0.243, 0.234, 1.58, 0.86)
STEM_PLACE_POSE = (0.811, -0.219, 0.47, 1.32, 0.86)
BAR_POSES_PLUS = ((0.884, -0.112, 0.021, 1.638, 0.933), (0.884, -0.121, 0.142, 1.55, 0.933))
BAR_POSES_MINUS = ((0.725, -0.35, 0.261, 1.638, 0.774), (0.725, -0.359, 0.381, 1.549, 0.774))

# 受け取り方ごとの設定
_SIDE = HandoverEnvCfg()
_TOP = HandoverTopEnvCfg()


def _block_cfg(path: str, color: tuple[float, float, float], pos) -> RigidObjectCfg:
    # キネマティックなブロックの設定を作成
    return RigidObjectCfg(
        prim_path=path,
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class RelayTEnvCfg(PickEnvCfg):
    # 両方の方策の行動を受け取る
    action_space = 10
    episode_length_s = 40.0

    # 青から始める
    block_starts = (BLUE_START,)
    block_cfg: RigidObjectCfg = _block_cfg("/World/envs/env_.*/Block", (0.1, 0.3, 0.9), BLUE_START)
    red_block_cfg: RigidObjectCfg = _block_cfg("/World/envs/env_.*/RedBlock", (0.9, 0.15, 0.1), RED_START)

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
    home_time = 1.4              # [s]

    # 置く動作の時間を設定 [s]
    place_hover_time = 1.5
    place_down_time = 1.0
    release_time = 2.8
    retreat_start = 3.0
    retreat_end = 3.8
    retreat_up = 0.06            # [m] 離したあと真上に上げる量
    settle_time = 0.6            # [s] 掴んだあと爪の間の決まった姿勢へ移す時間
    sag_gain = 0.1
    sag_limit = 0.03             # [m]

    # D435i を設定（マウントは常に配置、カメラは認識や撮影のときだけ）
    use_camera = False
    camera_depth = False
    perception_wait = 0.0        # [s] 認識のあいだ供給側を止める時間


class RelayTEnv(PickEnv):
    # 青を立て、赤を上に載せて T 字を作る通しの環境（環境 1 つ）
    cfg: RelayTEnvCfg

    def __init__(self, cfg: RelayTEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self.num_envs == 1
        n = 1
        dev = self.device

        # 供給側の行動バッファを 5 次元に設定
        self._actions = torch.zeros(n, GRIPPER, device=dev)
        self._prev_actions = torch.zeros(n, GRIPPER, device=dev)

        # 組立側のリンクと可動域を取得
        self._recv_hand_id = self.builder.find_bodies(cfg.hand_body_name)[0][0]
        limits = self.builder.data.soft_joint_pos_limits[:, self._builder_joint_ids]
        self._recv_lower = limits[..., 0]
        self._recv_upper = limits[..., 1]
        self._recv_rest = torch.tensor([[RECEIVER_INIT_POS[k] for k in cfg.joint_names]], device=dev)
        self._feeder_rest = torch.tensor([FEEDER_INIT_POS[k] for k in cfg.joint_names[:GRIPPER]], device=dev)
        self._arm_r = BatchArm(cfg.urdf_path, cfg.hand_body_name, list(cfg.joint_names[:GRIPPER]), _SIDE.tcp_offset, dev)
        self._r_tcp_offset = torch.tensor([_SIDE.tcp_offset], device=dev)
        self._up = torch.tensor([[0.0, 0.0, 1.0]], device=dev)

        # 段階ごとの設定（0: 青を立てる、1: 赤を載せる）
        self._stages = [
            {"rc": _TOP, "center": STAND_HANDOVER_CENTER, "pose": LOW_PRESENT_POSE, "via": True, "present": 1.6,
             "place": STAND_STEM_CENTER, "hover": 0.035,
             "start": {1: (STEM_HOVER_POSE, STEM_PLACE_POSE), -1: (STEM_HOVER_POSE, STEM_PLACE_POSE)}},
            {"rc": _SIDE, "center": HANDOVER_CENTER, "pose": PRESENT_POSE, "via": False, "present": 2.0,
             "place": STAND_BAR_CENTER, "hover": 0.012,
             "start": {1: BAR_POSES_PLUS, -1: BAR_POSES_MINUS}},
        ]
        self.stage = 0
        self.perceived: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.place_err: list[float] = []
        self._r_actions = torch.zeros(n, GRIPPER, device=dev)
        self._r_targets = self._recv_rest.clone()
        self._r_phase = torch.zeros(n, 2, device=dev)
        self.t_success = False
        self._reset_stage_state(home_from=self._recv_rest[:, :GRIPPER])
        self._set_stage(0)
        self._update_recv_frames()

    # ------------------------------------------------------------------
    # シーン
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # ロボットと 2 つのブロックを作成
        self.feeder = Articulation(self.cfg.feeder_cfg)
        self.builder = Articulation(self.cfg.builder_cfg)
        self.block = RigidObject(self.cfg.block_cfg)
        self.block_red = RigidObject(self.cfg.red_block_cfg)
        self.block_blue = self.block

        # IK 版と同じ位置に D435i のマウントとカメラを配置
        mount = sim_utils.UsdFileCfg(usd_path=D435I_MOUNT_USD)
        mount.func("/World/envs/env_0/D435iMount", mount, translation=D435I_MOUNT_POS)
        self.d435i = None
        if self.cfg.use_camera:
            cam_cfg = make_d435i_camera_cfg(depth=self.cfg.camera_depth).replace(prim_path="/World/envs/env_.*/D435iColor")
            self.d435i = Camera(cam_cfg)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["feeder"] = self.feeder
        self.scene.articulations["builder"] = self.builder
        self.scene.rigid_objects["block"] = self.block
        self.scene.rigid_objects["block_red"] = self.block_red
        if self.d435i is not None:
            self.scene.sensors["d435i_color"] = self.d435i

    # ------------------------------------------------------------------
    # 段階の切替
    # ------------------------------------------------------------------

    def _block_pose(self):
        # 掴む前は認識した姿勢を使用（認識していなければ正解値）
        if hasattr(self, "perceived") and self.stage in self.perceived and not bool(self._attached[0]) and not self._handed:
            return self.perceived[self.stage]
        return self.block.data.root_pos_w, self.block.data.root_quat_w

    def apply_perception(self):
        # 認識した姿勢で掴む端と参照軌道を計算し直す
        if self.stage not in self.perceived:
            return
        pos, quat = self.perceived[self.stage]
        axis = quat_apply(quat, self._x_axis)
        base = self.feeder.data.root_pos_w
        plus = torch.norm((pos + axis * self.cfg.grip_offset - base)[:, :2], dim=-1)
        minus = torch.norm((pos - axis * self.cfg.grip_offset - base)[:, :2], dim=-1)
        self._grip_side[:] = torch.where(plus < minus, 1.0, -1.0)
        self._fit_reference(torch.zeros(1, dtype=torch.long, device=self.device), pos, axis)
        self._update_frames()

    def _now(self) -> float:
        return float(self.episode_length_buf[0]) * self.step_dt

    def _set_stage(self, k: int):
        # 対象のブロックと差し出す位置を切替
        st = self._stages[k]
        self.stage = k
        self.rc = st["rc"]
        self.block = self.block_blue if k == 0 else self.block_red
        self._present_center[:] = torch.tensor(st["center"], device=self.device)
        self._present_pose[:] = torch.tensor(st["pose"], device=self.device)
        self.cfg.present_via_hover = st["via"]
        self.cfg.ref_present_time = st["present"]
        self._q_pre = torch.tensor([self.rc.pre_pose], device=self.device)
        self._q_recv = torch.tensor([self.rc.recv_pose], device=self.device)
        self._r_window = torch.tensor(self.rc.grasp_window, device=self.device)

    def _reset_stage_state(self, home_from: torch.Tensor):
        # 受け取る側の状態を初期化
        self._active = False
        self._handed = False
        self._placed = False
        self._start_time = 0.0
        self._hand_time = 0.0
        self._home_start = self._now() if hasattr(self, "episode_length_buf") else 0.0
        self._home_from = home_from.clone()
        self._home_to = self._recv_rest[:, :GRIPPER].clone()
        self._waiting = False
        self._grasp_abs = 0.0
        self._slip = torch.zeros(1, 3, device=self.device)
        self._r_ref = home_from.clone()
        self._r_ref_start = home_from.clone()
        self._r_ref_from = home_from.clone()
        self._r_grasp_steps = 0
        self._r_in_jaws = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._sag = torch.zeros(1, 3, device=self.device)
        self._q_ph = None
        self._q_pl = None
        self._feeder_from = self._feeder_rest.unsqueeze(0).clone()

    def _start_red(self):
        # 赤の掴む動作を開始
        dev = self.device
        ids = torch.zeros(1, dtype=torch.long, device=dev)
        now = self._now()
        self._set_stage(1)
        pos, quat = self._block_pose()
        pos = pos.clone()
        axis = quat_apply(quat, self._x_axis)
        base = self.feeder.data.root_pos_w
        plus = torch.norm((pos + axis * self.cfg.grip_offset - base)[:, :2], dim=-1)
        minus = torch.norm((pos - axis * self.cfg.grip_offset - base)[:, :2], dim=-1)
        self._grip_side[:] = torch.where(plus < minus, 1.0, -1.0)
        self._choice[:] = 1
        q = self.feeder.data.joint_pos[:, self._joint_ids]
        self._ref_start[:] = q[:, :GRIPPER]
        self._ref[:] = q[:, :GRIPPER]
        self._ref_from[:] = q[:, :GRIPPER]
        self._t0[:] = now
        self._grasp_time[:] = 0.0
        self._residual[:] = 0.0
        self._attached[:] = False
        self._grasped[:] = False
        self._just_grasped[:] = False
        self._squeeze_steps[:] = 0
        self._present_steps[:] = 0
        self._succeeded[:] = False
        self._penetrating[:] = False
        self._fit_reference(ids, pos, axis)
        self._reset_stage_state(home_from=self._r_ref.clone())

    # ------------------------------------------------------------------
    # 受け取る側の座標
    # ------------------------------------------------------------------

    def _update_recv_frames(self):
        rc = self.rc
        top = rc.grasp_mode == "top"
        hand_pos = self.builder.data.body_pos_w[:, self._recv_hand_id]
        hand_quat = self.builder.data.body_quat_w[:, self._recv_hand_id]
        self._r_tcp_pos, self._r_tcp_quat = combine_frame_transforms(hand_pos, hand_quat, self._r_tcp_offset, self._identity_quat)
        self._r_tcp_y = quat_apply(self._r_tcp_quat, self._y_axis)
        self._r_approach = -quat_apply(self._r_tcp_quat, self._z_axis)

        # 供給側が掴んでいない方の端を把持点に設定
        block_pos = self.block.data.root_pos_w
        block_quat = self.block.data.root_quat_w
        self._r_axis = quat_apply(block_quat, self._x_axis) * self._grip_side.unsqueeze(-1)
        depth = BLOCK_SIZE[0] / 2 - rc.top_grasp_depth if top else rc.grip_offset
        self._r_grasp = block_pos - self._r_axis * depth
        grasp_in_tcp = quat_apply(quat_conjugate(self._r_tcp_quat), self._r_grasp - self._r_tcp_pos)
        inside = bool(torch.all(torch.abs(grasp_in_tcp) < self._r_window))
        if top:
            by = quat_apply(block_quat, self._y_axis)
            bz = quat_apply(block_quat, self._z_axis)
            cy = torch.sum(self._r_tcp_y * by, dim=-1)
            cz = torch.sum(self._r_tcp_y * bz, dim=-1)
            self._jaw_side = by if bool(cy.abs() >= cz.abs()) else bz
            facing = float(torch.sum(self._r_approach * self._r_axis, dim=-1))
            aligned = facing > rc.grasp_align_cos and float(torch.maximum(cy.abs(), cz.abs())) > rc.grasp_align_cos
            self._r_align_signed = torch.ones(1, device=self.device)
        else:
            self._r_align_signed = torch.sum(self._r_tcp_y * self._r_axis, dim=-1)
            aligned = float(self._r_align_signed.abs()) > rc.grasp_align_cos
        self._r_in_jaws[:] = self._active and not self._handed and inside and aligned

    def _receive_reference(self):
        # 受け取る姿勢を更新し、参照軌道を計算
        rc = self.rc
        top = rc.grasp_mode == "top"
        root_pos = self.builder.data.root_pos_w
        root_quat = self.builder.data.root_quat_w
        inv = quat_conjugate(root_quat)
        grasp_b, _ = subtract_frame_transforms(root_pos, root_quat, self._r_grasp)
        axis_b = quat_apply(inv, self._r_axis)
        flat = grasp_b.clone()
        flat[:, 2] = 0.0
        back = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        q_pre, q_recv = self._q_pre, self._q_recv
        if top:
            pre_b = grasp_b - back * rc.pre_distance + self._up * rc.top_pre_up
            side_b = quat_apply(inv, self._jaw_side)
            for _ in range(rc.ik_iters):
                q_pre = self._arm_r.step(q_pre, pre_b, side_b, -axis_b, rot_weight=rc.top_rot_weight)
                q_recv = self._arm_r.step(q_recv, grasp_b, side_b, -axis_b, rot_weight=rc.top_rot_weight)
        else:
            pre_b = grasp_b - back * rc.pre_distance
            for _ in range(rc.ik_iters):
                q_pre = self._arm_r.step(q_pre, pre_b, axis_b)
                q_recv = self._arm_r.step(q_recv, grasp_b, axis_b)
        lower = self._recv_lower[:, :GRIPPER]
        upper = self._recv_upper[:, :GRIPPER]
        self._q_pre = torch.clamp(q_pre, lower, upper)
        self._q_recv = torch.clamp(q_recv, lower, upper)
        t = self._now() - self._start_time
        a = _smoothstep(torch.tensor(t / rc.ref_pre_time))
        b = _smoothstep(torch.tensor((t - rc.ref_pre_time) / rc.ref_recv_time))
        self._r_phase[:, 0] = min(max(t / (rc.ref_pre_time + rc.ref_recv_time), 0.0), 1.0)
        self._r_phase[:, 1] = 0.0
        return self._r_ref_start + (self._q_pre - self._r_ref_start) * a + (self._q_recv - self._q_pre) * b

    def _place_reference(self):
        # 置く位置へ運び、離して上へ退避する参照軌道を計算
        cfg = self.cfg
        rc = self.rc
        top = rc.grasp_mode == "top"
        tp = self._now() - self._hand_time
        root_pos = self.builder.data.root_pos_w
        root_quat = self.builder.data.root_quat_w
        inv = quat_conjugate(root_quat)
        center = torch.tensor([self._stages[self.stage]["place"]], device=self.device) + self.scene.env_origins
        if self._q_ph is None:
            sign = 1 if top or float(self._block_in_recv_pos[0, 1]) >= 0 else -1
            ph, pl = self._stages[self.stage]["start"][sign]
            self._q_ph = torch.tensor([ph], device=self.device)
            self._q_pl = torch.tensor([pl], device=self.device)

        # 置いたときのブロックの位置ずれを積算して補正
        if not self._placed and tp > cfg.place_hover_time:
            err = center - self.block.data.root_pos_w
            self._sag = (self._sag + cfg.sag_gain * err).clamp(-cfg.sag_limit, cfg.sag_limit)
        if top:
            tcp = center + self._up * rc.top_grasp_depth + self._sag
            target_y = None
        else:
            # 赤は掴んだ端から中心までのずれを考慮して横向きに置く
            x_w = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
            _, rot = self._arm_r.fk(self._q_pl)
            y_w = quat_apply(root_quat, rot[:, :, 1])
            s = 1.0 if float(torch.sum(y_w * x_w)) >= 0 else -1.0
            sign = float(self._block_in_recv_pos[0, 1]) / rc.grip_offset
            tcp = center - x_w * (sign * rc.grip_offset * s) + self._sag
            target_y = quat_apply(inv, x_w)
        tcp_b, _ = subtract_frame_transforms(root_pos, root_quat, tcp)
        hover_b = tcp_b + self._up * self._stages[self.stage]["hover"]
        for _ in range(3):
            self._q_ph = self._arm_r.step(self._q_ph, hover_b, target_y, self._up, rot_weight=0.1)
            self._q_pl = self._arm_r.step(self._q_pl, tcp_b, target_y, self._up, rot_weight=0.1)
        lower = self._recv_lower[:, :GRIPPER]
        upper = self._recv_upper[:, :GRIPPER]
        self._q_ph = torch.clamp(self._q_ph, lower, upper)
        self._q_pl = torch.clamp(self._q_pl, lower, upper)

        if self._placed:
            # 離したあとは真上に上げる
            c = float(_smoothstep(torch.tensor((tp - cfg.retreat_start) / (cfg.retreat_end - cfg.retreat_start))))
            return self._q_rel + (self._q_up - self._q_rel) * c
        a = float(_smoothstep(torch.tensor(tp / cfg.place_hover_time)))
        b = float(_smoothstep(torch.tensor((tp - cfg.place_hover_time) / cfg.place_down_time)))
        return self._r_ref_from + (self._q_ph - self._r_ref_from) * a + (self._q_pl - self._q_ph) * b

    def _release(self):
        # ブロックを離して置き場所に合わせる
        center = torch.tensor([self._stages[self.stage]["place"]], device=self.device) + self.scene.env_origins
        quat = self.block.data.root_quat_w
        axis = quat_apply(quat, self._x_axis)
        # 離す直前の位置の誤差を記録
        self.place_err.append(float(torch.norm(self.block.data.root_pos_w - center)))
        if self.stage == 0:
            # 縦に立て、向きは今の回転を保つ
            x = torch.tensor([[0.0, 0.0, 1.0]], device=self.device) * (1.0 if float(axis[0, 2]) >= 0 else -1.0)
            y = quat_apply(quat, self._y_axis)
            y[:, 2] = 0.0
            if float(y.norm()) < 1e-3:
                y = quat_apply(quat, self._z_axis)
                y[:, 2] = 0.0
        else:
            # 横棒を x 方向にそろえる
            x = torch.tensor([[1.0, 0.0, 0.0]], device=self.device) * (1.0 if float(axis[0, 0]) >= 0 else -1.0)
            y = torch.cross(torch.tensor([[0.0, 0.0, 1.0]], device=self.device), x, dim=-1)
        y = y / y.norm(dim=-1, keepdim=True)
        z = torch.cross(x, y, dim=-1)
        rot = torch.stack([x, y, z], dim=-1)
        self.block.write_root_pose_to_sim(torch.cat([center, quat_from_matrix(rot)], dim=-1))

        # 真上に上げる姿勢を計算（手先の位置だけを合わせる）
        self._q_rel = self._r_ref.clone()
        pos, _ = self._arm_r.fk(self._q_rel)
        target = pos + self._up * self.cfg.retreat_up
        q = self._q_rel.clone()
        for _ in range(40):
            q = self._arm_r.step(q, target)
        self._q_up = torch.clamp(q, self._recv_lower[:, :GRIPPER], self._recv_upper[:, :GRIPPER])
        self._placed = True

    # ------------------------------------------------------------------
    # 行動
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        cfg = self.cfg
        rc = self.rc
        now = self._now()

        # 供給側の関節目標を更新（認識のあいだは止める）
        pick_actions = actions[:, :GRIPPER]
        if now < float(self._t0[0]):
            pick_actions = torch.zeros_like(pick_actions)
        super()._pre_physics_step(pick_actions)
        if self._handed:
            s = float(_smoothstep(torch.tensor((now - self._hand_time) / cfg.feeder_return_time)))
            self._targets[:, :GRIPPER] = self._feeder_from + (self._feeder_rest - self._feeder_from) * s
            self._targets[:, GRIPPER] = cfg.gripper_open_angle
        if self.stage >= 2:
            self._targets[:, :GRIPPER] = self._feeder_rest
            self._targets[:, GRIPPER] = cfg.gripper_open_angle

        # 受け取る側の関節目標を更新
        self._r_actions[:] = actions[:, GRIPPER:].clamp(-1.0, 1.0)
        self._update_recv_frames()
        if self._handed:
            ref = self._place_reference()
            residual = torch.zeros_like(ref)
        elif self._active:
            ref = self._receive_reference()
            residual = self._r_actions * rc.residual_limit
        else:
            s = float(_smoothstep(torch.tensor((now - self._home_start) / cfg.home_time)))
            ref = self._home_from + (self._home_to - self._home_from) * s
            residual = torch.zeros_like(ref)
        self._r_ref = ref
        self._r_targets[:, :GRIPPER] = torch.clamp(ref + residual, self._recv_lower[:, :GRIPPER], self._recv_upper[:, :GRIPPER])
        close = bool(self._r_in_jaws[0]) or (self._handed and not self._placed)
        self._r_targets[:, GRIPPER] = rc.gripper_close_target if close else rc.gripper_open_angle

    def _lift_above_floor(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        # 持っているブロックが台や立てた青に入り込まないよう、爪の中で上にずらす
        signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                             dtype=torch.float32, device=self.device)
        half = torch.tensor(BLOCK_SIZE, device=self.device) * 0.5
        corners = pos + quat_apply(quat.expand(8, 4), signs * half)
        floor = torch.full((8,), float(self.scene.env_origins[0, 2]), device=self.device)
        if self.stage == 1:
            # 立てた青の上面を床とみなす
            blue_pos = self.block_blue.data.root_pos_w
            blue_quat = self.block_blue.data.root_quat_w
            local = quat_apply(quat_conjugate(blue_quat).expand(8, 4), corners - blue_pos)
            on_top = (local[:, 1].abs() < half[1]) & (local[:, 2].abs() < half[2])
            floor = torch.where(on_top, blue_pos[0, 2] + half[0], floor)
        lift = float((floor - corners[:, 2]).clamp(min=0.0).max())
        if lift > 0.0:
            pos = pos.clone()
            pos[:, 2] += lift
        return pos

    def _apply_action(self):
        self.feeder.set_joint_position_target(self._targets, joint_ids=self._joint_ids)
        self.builder.set_joint_position_target(self._r_targets, joint_ids=self._builder_joint_ids)

        # 供給側が持っているブロックを固定
        if bool(self._attached[0]) and not self._handed:
            tcp_pos, tcp_quat = self._tcp_pose()
            rel_pos, rel_quat = self._block_in_tcp_pos, self._block_in_tcp_quat
            if self.cfg.keep_grasp_pose and hasattr(self, "_canon_pos"):
                # 掴んだときの姿勢から、爪の間の決まった姿勢へ少しずつ移す
                u = float(_smoothstep(torch.tensor((self._now() - self._grasp_abs) / self.cfg.settle_time)))
                rel_pos = self._block_in_tcp_pos * (1.0 - u) + self._canon_pos * u
                canon = self._canon_quat * torch.where(torch.sum(self._canon_quat * rel_quat, dim=-1, keepdim=True) < 0, -1.0, 1.0)
                rel_quat = rel_quat * (1.0 - u) + canon * u
                rel_quat = rel_quat / rel_quat.norm(dim=-1, keepdim=True)
            pos, quat = combine_frame_transforms(tcp_pos, tcp_quat, rel_pos + self._slip, rel_quat)
            lifted = self._lift_above_floor(pos, quat)
            if not torch.equal(lifted, pos):
                self._slip = self._slip + quat_apply(quat_conjugate(tcp_quat), lifted - pos)
            self.block.write_root_pose_to_sim(torch.cat([lifted, quat], dim=-1))

        # 受け取る側が持っているブロックを固定
        if self._handed and not self._placed:
            hand_pos = self.builder.data.body_pos_w[:, self._recv_hand_id]
            hand_quat = self.builder.data.body_quat_w[:, self._recv_hand_id]
            tcp_pos, tcp_quat = combine_frame_transforms(hand_pos, hand_quat, self._r_tcp_offset, self._identity_quat)
            pos, quat = combine_frame_transforms(tcp_pos, tcp_quat, self._block_in_recv_pos, self._block_in_recv_quat)
            lifted = self._lift_above_floor(pos, quat)
            if not torch.equal(lifted, pos):
                self._block_in_recv_pos = subtract_frame_transforms(tcp_pos, tcp_quat, lifted, quat)[0]
            self.block.write_root_pose_to_sim(torch.cat([lifted, quat], dim=-1))

    # ------------------------------------------------------------------
    # 状態の更新
    # ------------------------------------------------------------------

    def _get_dones(self):
        cfg = self.cfg
        rc = self.rc
        _, time_out = super()._get_dones()
        if self._handed or self.stage >= 2:
            self._attached[:] = False
        now = self._now()
        if bool(self._just_grasped[0]):
            self._grasp_abs = now
            self._slip = torch.zeros(1, 3, device=self.device)

        # 青は低い位置で受け渡すので、供給側が掴んだら組立側を受け取る手前で待たせる
        if self.stage == 0 and bool(self._attached[0]) and not self._active and not self._waiting:
            self._waiting = True
            self._home_from = self._r_ref.clone()
            self._home_to = torch.tensor([self.rc.pre_pose], device=self.device)
            self._home_start = now
        t_pick = now - float(self._t0[0])

        # 差し出したら受け取りを開始（組立側が休息姿勢に戻ってから）
        home_done = now - self._home_start >= cfg.home_time
        if self.stage < 2 and bool(self._attached[0]) and not self._active and home_done and (
            int(self._present_steps[0]) >= cfg.start_present_steps
            or t_pick - float(self._grasp_time[0]) > cfg.start_timeout
        ):
            self._active = True
            self._start_time = now
            self._r_ref_start = self.builder.data.joint_pos[:, self._builder_joint_ids[:GRIPPER]].clone()

        # 受け渡しを判定
        self._update_recv_frames()
        grip = float(self.builder.data.joint_pos[0, self._builder_joint_ids[GRIPPER]])
        squeezing = bool(self._r_in_jaws[0]) and grip < rc.gripper_closed_max
        self._r_grasp_steps = self._r_grasp_steps + 1 if squeezing else 0
        if not self._handed and self._r_grasp_steps >= rc.release_steps:
            if rc.grasp_mode == "top":
                self._block_in_recv_pos = torch.tensor([[0.0, 0.0, -rc.top_grasp_depth]], device=self.device)
                self._block_in_recv_quat = torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]], device=self.device)
            else:
                sign = 1.0 if float(self._r_align_signed[0]) >= 0 else -1.0
                self._block_in_recv_pos = torch.tensor([[rc.grasp_center_x, sign * rc.grip_offset, 0.0]], device=self.device)
                self._block_in_recv_quat = torch.tensor([[0.7071068, 0.0, 0.0, sign * 0.7071068]], device=self.device)
            self._handed = True
            self._hand_time = now
            self._r_ref_from = self._r_ref.clone()
            self._feeder_from = self.feeder.data.joint_pos[:, self._joint_ids[:GRIPPER]].clone()
            self._attached[:] = False

        # 置いて退避したら次の段階へ
        if self._handed:
            tp = now - self._hand_time
            if not self._placed and tp >= cfg.release_time:
                self._release()
            if self._placed and tp >= cfg.retreat_end:
                if self.stage == 0:
                    self._start_red()
                else:
                    self.stage = 2
                    self.t_success = True
                    self._reset_stage_state(home_from=self._r_ref.clone())
        return torch.zeros_like(time_out), time_out

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(1, device=self.device)

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
        tcp_dir = self._r_approach if self.rc.grasp_mode == "top" else self._r_tcp_y
        obs["receiver"] = torch.cat(
            [
                joint_pos - self._recv_rest,
                joint_vel,
                self._r_actions,
                tcp_b,
                grasp_b,
                grasp_b - tcp_b,
                quat_apply(inv, self._r_axis),
                quat_apply(inv, tcp_dir),
                torch.full((1, 1), float(self._handed), device=self.device),
                self._r_phase,
            ],
            dim=-1,
        )
        return obs

    # ------------------------------------------------------------------
    # リセット
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.feeder._ALL_INDICES
        if hasattr(self, "_stages"):
            self._set_stage(0)
        super()._reset_idx(env_ids)

        # 赤を台に置く
        dev = self.device
        pos = torch.tensor([RED_START], device=dev)
        pos[:, :2] += (torch.rand(1, 2, device=dev) * 2.0 - 1.0) * RANDOM_XY
        pos += self.scene.env_origins[env_ids]
        yaw = math.radians(self.cfg.block_start_yaw_deg) + (float(torch.rand(1)) * 2.0 - 1.0) * math.radians(RANDOM_YAW_DEG)
        quat = quat_from_angle_axis(torch.tensor([yaw], device=dev), self._z_axis[env_ids])
        self.block_red.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

        if hasattr(self, "_stages"):
            self._t0[:] = self.cfg.perception_wait
            self.perceived = {}
            self._r_targets[:] = self._recv_rest
            self.t_success = False
            self.place_err = []
            self._reset_stage_state(home_from=self._recv_rest[:, :GRIPPER])
