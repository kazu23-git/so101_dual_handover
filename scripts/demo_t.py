"""強化学習の方策で受け渡し、青を立てて赤を載せる T 字の組み立て"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--pick_low", type=str, default="latest", help="青を低い位置に差し出す方策")
parser.add_argument("--pick", type=str, default="latest", help="赤を差し出す方策")
parser.add_argument("--handover_top", type=str, default="latest", help="上端を真上からつまむ方策")
parser.add_argument("--handover", type=str, default="latest", help="上端を横から掴む方策")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--video", action="store_true", help="各回を動画で保存")
parser.add_argument("--video_dir", type=str, default="outputs/t_video")
parser.add_argument("--view", type=str, default="overview", help="視点（overview, front, back, top, handover, t_close）")
parser.add_argument("--seed", type=int, default=0, help="ブロックの置き方を決める乱数の種")
parser.add_argument("--perception", action="store_true", help="D435i の画像と物体検出でブロックの位置を推定")
parser.add_argument("--depth_noise", action="store_true", help="距離比例の深度ノイズを付加")
parser.add_argument("--det_threshold", type=float, default=0.1, help="検出スコアの閾値")
parser.add_argument("--image_dir", type=str, default="outputs/t_perception", help="検出画像の保存先")
parser.add_argument("--side_camera", action="store_true", help="D435i のカラー画像を各回連番で保存")
parser.add_argument("--side_dir", type=str, default="outputs/t_side_camera")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim を起動
if args.video or args.perception or args.side_camera:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import os
import shutil
import subprocess

import torch
from PIL import Image

from so101_handover import t_task_cfg as T
from so101_handover.perception import BlockDetector, estimate_block_pose, save_detection_image, yaw_to_quat
from so101_handover.relay_t_env import RelayTEnv, RelayTEnvCfg

# 視点を定義（環境 0 の原点からの位置 [m]、eye と lookat）
VIEWS = {
    "overview": ((0.55, -0.6, 0.45), (0.05, -0.05, 0.08)),
    "front": ((0.75, 0.0, 0.3), (0.0, 0.0, 0.1)),
    "back": ((0.25, 0.75, 0.45), (0.0, -0.05, 0.08)),
    "top": ((0.05, 0.0, 0.95), (0.0, 0.0, 0.0)),
    "handover": ((0.3, -0.22, 0.22), (0.0, -0.03, 0.1)),
    "t_close": ((0.38, -0.32, 0.2), (0.12, -0.14, 0.06)),
}

LOG_DIRS = {
    "pick_low": "logs/rsl_rl/so101_pick_low",
    "pick": "logs/rsl_rl/so101_pick_present",
    "handover_top": "logs/rsl_rl/so101_handover_top",
    "handover": "logs/rsl_rl/so101_handover",
}


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


def perceive(env, detector, name: str, ep: int):
    # D435i の画像からブロックの位置と向きを推定
    camera = env.d435i
    if camera.data.intrinsic_matrices[0, 0, 0].item() == 0.0:
        camera.reset()
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].clone()
    if args.depth_noise:
        depth = depth + torch.randn_like(depth) * depth * T.DEPTH_NOISE_REL
    depth[depth < T.D435I_DEPTH_MIN] = 0.0
    intrinsic = camera.data.intrinsic_matrices[0]
    cam_pos = camera.data.pos_w[0]
    cam_quat = camera.data.quat_w_ros[0]
    query = T.BLOCK_QUERIES[name]
    path = os.path.join(args.image_dir, f"ep{ep}_detect_{query.replace(' ', '_')}.png")
    found = detector.detect(rgb, query)
    est = None
    if found is not None:
        est = estimate_block_pose(found[0], depth, intrinsic, cam_pos, cam_quat, T.TOP_Z_RANGE, T.BLOCK_HALF_HEIGHT,
                                  T.BLOCK_SIZE[0], T.OCCLUSION_TOL)
    if est is None:
        save_detection_image(path, rgb, found[0] if found else None, found[1] if found else 0.0, query, None, 0.0,
                             intrinsic, cam_pos, cam_quat, 0.0)
        print(f"[認識失敗] {name}: 正解値で続行", flush=True)
        return None
    center, yaw, n_points, _ = est
    save_detection_image(path, rgb, found[0], found[1], query,
                         center + torch.tensor([0.0, 0.0, T.BLOCK_HALF_HEIGHT], device=center.device),
                         yaw, intrinsic, cam_pos, cam_quat, T.BLOCK_SIZE[0] / 2)

    # 正解値との誤差を表示
    block = env.block_blue if name == "青" else env.block_red
    true_pos = block.data.root_pos_w[0]
    true_axis = quat_axis(block.data.root_quat_w)[0]
    yaw_err = abs(math.remainder(yaw - math.atan2(true_axis[1].item(), true_axis[0].item()), math.pi))
    pos_err = (center - true_pos)[:2].norm().item()
    print(f"[認識] {name}: スコア {found[1]:.2f}、点 {n_points} 個、位置誤差 {pos_err * 1000:.1f} mm、"
          f"向き誤差 {math.degrees(yaw_err):.1f} deg", flush=True)
    return center.unsqueeze(0), yaw_to_quat(yaw, center.device)


def block_offset(env, block, start) -> str:
    # 基準の置き場所からのずれを文字列に変換
    pos = block.data.root_pos_w[0] - env.scene.env_origins[0]
    axis = quat_axis(block.data.root_quat_w)[0]
    yaw = math.degrees(math.atan2(axis[1].item(), axis[0].item()))
    yaw = yaw % 180.0 - 90.0
    return f"x {(pos[0].item() - start[0]) * 1000:+.0f} mm、y {(pos[1].item() - start[1]) * 1000:+.0f} mm、向き {yaw:+.0f} deg"


def to_mp4(frame_dir: str, out: str):
    # 連番画像を動画に変換
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(frame_dir, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
    print(f"[video] {out}", flush=True)


def quat_axis(quat: torch.Tensor) -> torch.Tensor:
    # クォータニオンの x 軸を計算
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)


def main():
    # 方策を読み込み
    paths = {k: getattr(args, k) for k in LOG_DIRS}
    for k, v in paths.items():
        if v == "latest":
            paths[k] = find_latest_checkpoint(LOG_DIRS[k])
        print(f"[{k}] {paths[k]}", flush=True)

    # 環境を作成
    cfg = RelayTEnvCfg()
    cfg.scene.num_envs = 1
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye, cfg.viewer.lookat = VIEWS[args.view]
    cfg.seed = args.seed
    cfg.viewer.resolution = (1920, 1080)
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**19
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.use_camera = args.perception or args.side_camera
    cfg.camera_depth = args.perception
    cfg.perception_wait = 0.5 if args.perception else 0.0
    env = RelayTEnv(cfg, render_mode="rgb_array" if args.video else None)
    base = env
    max_steps = int(cfg.episode_length_s / base.step_dt) - 2
    actors = {k: load_actor(v, base.device) for k, v in paths.items()}
    detector = None
    if args.perception:
        detector = BlockDetector(device=str(base.device), threshold=args.det_threshold)
        os.makedirs(args.image_dir, exist_ok=True)

    # 組み立てを実行
    results = []
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    for ep in range(args.episodes):
        events = {}
        done_at = None
        perceived = not args.perception
        frame = 0
        view_dir = os.path.join(args.video_dir, f"{args.view}_ep{ep}")
        side_dir = os.path.join(args.side_dir, f"ep{ep}")
        for d, on in ((view_dir, args.video), (side_dir, args.side_camera)):
            if on:
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d)
        for step in range(max_steps):
            if not simulation_app.is_running():
                break

            # ブロックの置き方を表示
            if step == 0:
                print(f"[{ep}] 青の置き方: {block_offset(base, base.block_blue, T.BLUE_START)}", flush=True)
                print(f"[{ep}] 赤の置き方: {block_offset(base, base.block_red, T.RED_START)}", flush=True)

            # 供給側が止まっているあいだに 2 つのブロックを認識
            if not perceived and (step + 1) * base.step_dt >= 0.3:
                perceived = True
                for k, name in ((0, "青"), (1, "赤")):
                    est = perceive(base, detector, name, ep)
                    if est is not None:
                        base.perceived[k] = est
                base.apply_perception()
                with torch.inference_mode():
                    obs = base._get_observations()

            with torch.inference_mode():
                if base.stage == 0:
                    a = torch.cat([actors["pick_low"](obs["policy"]), actors["handover_top"](obs["receiver"])], dim=-1)
                else:
                    a = torch.cat([actors["pick"](obs["policy"]), actors["handover"](obs["receiver"])], dim=-1)
                obs, _, _, _, _ = env.step(a)

            # 視点の画像と D435i の画像を保存
            if args.video:
                Image.fromarray(base.render()).save(os.path.join(view_dir, f"{frame:05d}.jpg"), quality=92)
            if args.side_camera:
                rgb = base.d435i.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
                Image.fromarray(rgb).save(os.path.join(side_dir, f"{frame:05d}.jpg"), quality=92)
            frame += 1

            # 進み具合を表示
            t = (step + 1) * base.step_dt
            name = ["青", "赤", "完了"][min(base.stage, 2)]
            for key, flag in ((f"{name}: 掴む", bool(base._grasped[0])), (f"{name}: 受け取り開始", base._active),
                              (f"{name}: 受け渡し", base._handed), (f"{name}: 置く", base._placed)):
                if base.stage < 2 and flag and key not in events:
                    events[key] = t
                    print(f"[{ep}] {t:5.2f} s {key}", flush=True)
            if base.t_success and done_at is None:
                done_at = t
                print(f"[{ep}] {t:5.2f} s T 字の完成", flush=True)
            if done_at is not None and t - done_at > 2.0:
                break

        # 離す直前の位置の誤差を表示
        errs = "、".join(f"{n} {e * 1000:.1f} mm" for n, e in zip(("青", "赤"), base.place_err))
        results.append((base.t_success, done_at))
        print(f"[{ep}] 成功 {base.t_success}、完成までの時間 {done_at}、離す直前の位置の誤差 {errs}", flush=True)
        if args.video:
            to_mp4(view_dir, view_dir + ".mp4")
        if args.side_camera:
            to_mp4(side_dir, os.path.join(args.side_dir, f"d435i_ep{ep}.mp4"))
        if ep < args.episodes - 1:
            obs, _ = env.reset()

    # 結果を表示
    n_ok = sum(1 for r in results if r[0])
    print("========== T 字の結果 ==========", flush=True)
    print(f"  成功 {n_ok} / {len(results)}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
