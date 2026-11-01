"""シーン確認用"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--base_distance", type=float, default=None, help="土台の間隔 [m]")
parser.add_argument("--giver_pose", type=float, nargs=6, default=None, help="渡す側の提示姿勢 [rad]")
parser.add_argument("--receiver_pose", type=float, nargs=6, default=None, help="受け取る側の初期姿勢 [rad]")
parser.add_argument("--no_noise", action="store_true", help="姿勢ノイズを無効化")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim を起動
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from so101_handover.handover_env import HandoverEnv
from so101_handover.handover_env_cfg import HandoverEnvCfg


def fmt(vec) -> str:
    # 座標を mm 単位の文字列に変換
    return "(" + ", ".join(f"{v * 1000:7.1f}" for v in vec.tolist()) + ") mm"


def main():
    # 設定を作成
    cfg = HandoverEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.base_distance is not None:
        cfg.base_distance = args.base_distance
    if args.giver_pose is not None:
        cfg.giver_present_pos = dict(zip(cfg.joint_names, args.giver_pose))
    if args.receiver_pose is not None:
        cfg.receiver_init_pos = dict(zip(cfg.joint_names, args.receiver_pose))
    if args.no_noise:
        cfg.giver_present_noise = (0.0,) * 6
        cfg.receiver_init_noise = 0.0
    cfg.__post_init__()

    # 環境を作成
    env = HandoverEnv(cfg)
    env.reset()

    # 設定値を表示
    print("[base_distance]", cfg.base_distance)
    print("[giver_pose]   ", [cfg.giver_present_pos[n] for n in cfg.joint_names])
    print("[receiver_pose]", [cfg.receiver_init_pos[n] for n in cfg.joint_names])
    print("[joints]", env.giver.joint_names)

    actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        env.step(actions)

        # 環境 0 の位置を表示
        if step % 60 == 0:
            origin = env.scene.env_origins[0]
            recv_base = env.receiver.data.root_pos_w[0]
            grasp_from_recv = env._grasp_pos[0] - recv_base
            reach_xy = torch.norm(grasp_from_recv[:2]).item()
            hand_gap = torch.norm(env._recv_tcp_pos[0] - env._giver_tcp_pos[0]).item()

            print(f"--- step {step}")
            print("  giver base    ", fmt(env.giver.data.root_pos_w[0] - origin))
            print("  receiver base ", fmt(recv_base - origin))
            print("  giver tcp     ", fmt(env._giver_tcp_pos[0] - origin))
            print("  receiver tcp  ", fmt(env._recv_tcp_pos[0] - origin))
            print("  block center  ", fmt(env._block_pos[0] - origin))
            print("  grasp point   ", fmt(env._grasp_pos[0] - origin))
            print(f"  把持点までの水平距離   {reach_xy * 1000:6.1f} mm")
            print(f"  両 TCP 間の距離        {hand_gap * 1000:6.1f} mm")
            print(f"  TCP と把持点の距離     {env._dist[0].item() * 1000:6.1f} mm")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
