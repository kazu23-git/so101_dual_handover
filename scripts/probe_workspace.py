"""SO-101 の到達範囲の調査"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=4.0, help="IK の実行時間 [s]")
parser.add_argument("--tol_mm", type=float, default=8.0, help="位置の許容誤差 [mm]")
parser.add_argument("--tol_deg", type=float, default=8.0, help="向きの許容誤差 [deg]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

# Isaac Sim を起動
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import PhysxCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from so101_handover import t_task_cfg as T
from so101_handover.ik import DlsIk, approach_dir, horizontal_grasp_quat, top_down_quat
from so101_handover.robot_cfg import make_so101_cfg

# 調査範囲を定義 [m]
RADII = [round(0.12 + 0.02 * i, 2) for i in range(12)]    # 0.12〜0.34
HEIGHTS = [round(0.02 + 0.02 * i, 2) for i in range(10)]  # 0.02〜0.20
MODES = ["真下向き", "水平向き"]


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    # 床と照明を配置
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    # ロボットを配置
    robot: ArticulationCfg = make_so101_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        pos=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        init_joint_pos=T.READY_POSE,
    )


def main():
    # 調べる組み合わせを作成
    cases = [(m, r, h) for m in MODES for h in HEIGHTS for r in RADII]
    num = len(cases)

    # シミュレーションを作成
    sim_cfg = sim_utils.SimulationCfg(
        dt=T.PHYSICS_DT, device=args.device, physx=PhysxCfg(gpu_max_rigid_patch_count=2**20)
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(ProbeSceneCfg(num_envs=num, env_spacing=0.9))
    sim.reset()
    dt = sim.get_physics_dt()
    scene.update(dt)

    robot = scene["robot"]
    ik = DlsIk(robot)
    dev = robot.device
    origins = scene.env_origins

    # 目標の位置と姿勢を作成
    target_pos = torch.zeros(num, 3, device=dev)
    target_quat = torch.zeros(num, 4, device=dev)
    forward = torch.tensor([[1.0, 0.0, 0.0]], device=dev)
    unit_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    for i, (mode, r, h) in enumerate(cases):
        target_pos[i] = origins[i] + torch.tensor([r, 0.0, h], device=dev)
        if mode == "真下向き":
            # 真下向きの姿勢を設定
            target_quat[i] = top_down_quat(forward)[0]
        else:
            # 水平向きの姿勢を設定
            target_quat[i] = horizontal_grasp_quat(forward, unit_quat)[0]

    # 初期姿勢で待機
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    for _ in range(int(0.5 / dt)):
        robot.set_joint_position_target(joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    # IK で目標へ移動
    grip = torch.full((num, 1), 0.3, device=dev)
    for _ in range(int(args.seconds / dt)):
        q_arm = ik.solve(target_pos, target_quat)
        robot.set_joint_position_target(q_arm, joint_ids=ik.arm_ids)
        robot.set_joint_position_target(grip, joint_ids=[ik.grip_id])
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    # 誤差を計算
    tcp_pos, tcp_quat = ik.tcp_pose()
    pos_err = (tcp_pos - target_pos).norm(dim=-1) * 1000
    dot = (approach_dir(tcp_quat) * approach_dir(target_quat)).sum(dim=-1).clamp(-1.0, 1.0)
    ang_err = torch.rad2deg(torch.acos(dot))

    # 結果の表を表示
    print()
    print(f"判定: ○ 位置 {args.tol_mm:.0f} mm 以内かつ向き {args.tol_deg:.0f} 度以内、"
          f"△ 位置のみ、数字は位置の誤差 [mm]")
    for mode in MODES:
        print()
        print(f"===== {mode}  横: 前方距離 [m] / 縦: TCP 高さ [m] =====")
        print("  高さ  " + " ".join(f"{r:5.2f}" for r in RADII))
        for h in reversed(HEIGHTS):
            cells = []
            for r in RADII:
                i = cases.index((mode, r, h))
                e, a = pos_err[i].item(), ang_err[i].item()
                if e <= args.tol_mm and a <= args.tol_deg:
                    cells.append("    ○")
                elif e <= args.tol_mm:
                    cells.append("    △")
                else:
                    cells.append(f"{min(e, 999):5.0f}")
            print(f"  {h:4.2f}  " + " ".join(cells))
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
