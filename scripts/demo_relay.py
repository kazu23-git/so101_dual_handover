"""供給側が掴んで差し出し、組立側が受け取る通しの再生"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--pick_checkpoint", type=str, default="latest", help="供給側の model_*.pt")
parser.add_argument("--handover_checkpoint", type=str, default="latest", help="受け取る側の model_*.pt")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=470)
parser.add_argument("--video", action="store_true", help="動画を保存")
parser.add_argument("--video_dir", type=str, default="outputs/relay_video")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim を起動
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

from so101_handover.relay_env import RelayEnv, RelayEnvCfg


def find_latest_checkpoint(root: str) -> str:
    # 最新の model_*.pt を検索
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for run in reversed(runs):
        names = [n for n in os.listdir(os.path.join(root, run)) if n.startswith("model_") and n.endswith(".pt")]
        if names:
            names.sort(key=lambda n: int(n[len("model_"):-len(".pt")]))
            return os.path.join(root, run, names[-1])
    raise FileNotFoundError(root)


def load_actor(path: str, device: str) -> torch.nn.Module:
    # チェックポイントから方策のネットワークを復元
    state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for i, j in enumerate(ids):
        weight = state[f"actor.{j}.weight"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[f"actor.{j}.bias"])
        layers.append(layer)
        if i < len(ids) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).to(device).eval()


def main():
    # チェックポイントを決定
    if args.pick_checkpoint == "latest":
        args.pick_checkpoint = find_latest_checkpoint("logs/rsl_rl/so101_pick_present")
    if args.handover_checkpoint == "latest":
        args.handover_checkpoint = find_latest_checkpoint("logs/rsl_rl/so101_handover")
    print(f"[pick] {args.pick_checkpoint}")
    print(f"[handover] {args.handover_checkpoint}")

    # 環境を作成
    cfg = RelayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # 環境 0 を中心にカメラを配置
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.6, -0.45, 0.4)
    cfg.viewer.lookat = (0.0, 0.0, 0.12)
    cfg.viewer.resolution = (1920, 1080)
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**19
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    env = RelayEnv(cfg, render_mode="rgb_array" if args.video else None)
    base = env
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, step_trigger=lambda step: step == 0,
            video_length=args.steps, disable_logger=True,
        )
        print(f"[video] {args.video_dir}")

    # 方策を読み込み
    pick = load_actor(args.pick_checkpoint, base.device)
    recv = load_actor(args.handover_checkpoint, base.device)

    # 再生を実行
    obs, _ = env.reset()
    events = {"grasp": None, "start": None, "hand": None, "success": None}
    for step in range(args.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = torch.cat([pick(obs["policy"]), recv(obs["receiver"])], dim=-1)
            obs, _, _, _, _ = env.step(actions)

        # 環境 0 の進み具合を表示
        t = step * base.step_dt
        for key, flag in (("grasp", base._grasped), ("start", base._active), ("hand", base._handed),
                          ("success", base.relay_success)):
            if events[key] is None and bool(flag[0]):
                events[key] = t
                print(f"[{t:5.2f} s] {key}", flush=True)

    # 結果を表示
    print("========== 通しの結果 ==========", flush=True)
    print(f"  掴んだ環境     {int(base._grasped.sum())} / {base.num_envs}", flush=True)
    print(f"  受け渡した環境 {int(base._handed.sum())} / {base.num_envs}", flush=True)
    print(f"  成功した環境   {int(base.relay_success.sum())} / {base.num_envs}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
