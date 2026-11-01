"""学習済み方策の再生"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["handover", "pick", "pick_low", "handover_top"], default="handover", help="再生する環境")
parser.add_argument("--checkpoint", type=str, default="latest", help="model_*.pt のパス")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--video", action="store_true", help="動画を保存")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--from_rest", action="store_true", help="pick を休息姿勢から開始")
parser.add_argument("--zero_action", action="store_true", help="行動 0 で再生")
parser.add_argument("--stochastic", action="store_true", help="学習時と同じく行動にノイズを加えて再生")
parser.add_argument("--diagnose", action="store_true", help="把持判定の診断を表示")
parser.add_argument("--receiver_gripper", type=float, default=None, help="受け取る側のグリッパ初期角 [rad]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

# Isaac Sim を起動
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from importlib import metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from so101_handover.tasks import TASKS


def find_latest_checkpoint(root: str) -> str:
    # 最新の model_*.pt を検索
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for run in reversed(runs):
        names = [n for n in os.listdir(os.path.join(root, run)) if n.startswith("model_") and n.endswith(".pt")]
        if names:
            names.sort(key=lambda n: int(n[len("model_"):-len(".pt")]))
            return os.path.join(root, run, names[-1])
    raise FileNotFoundError(f"{root} にチェックポイントがない")


def main():
    task_id, env_cfg_cls, agent_cfg_cls = TASKS[args.task]
    agent_cfg = agent_cfg_cls()

    # チェックポイントを決定
    if args.checkpoint == "latest":
        args.checkpoint = find_latest_checkpoint(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[checkpoint] {args.checkpoint}")

    # 設定を作成
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.eye = (0.7, 0.7, 0.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.15)
    # GPU の接触バッファを縮小
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**19
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    if args.from_rest and args.task.startswith("pick"):
        env_cfg.hover_start_prob = 0.0
        env_cfg.attached_start_prob = 0.0
    if args.receiver_gripper is not None and args.task.startswith("handover"):
        env_cfg.receiver_init_pos = {**env_cfg.receiver_init_pos, "gripper": args.receiver_gripper}

    # rsl-rl の版に合わせて設定を変換
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except ImportError:
        pass

    # 環境を作成
    render_mode = "rgb_array" if args.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)

    # 動画の保存を設定
    if args.video:
        video_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[video] {video_dir}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 方策を読み込み
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    actor = getattr(runner.alg, "policy", None) or runner.alg.actor_critic
    action_std = (actor.std if hasattr(actor, "std") else actor.log_std.exp()).detach()
    if args.stochastic:
        print(f"[stochastic] std {action_std.mean().item():.3f}")

    # 観測を取得
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # 再生を実行
    base_env = env.unwrapped
    num_success = 0
    num_drop = 0
    diag = None
    if args.diagnose:
        diag = HandoverDiag(base_env) if args.task.startswith("handover") else PickDiag(base_env)
    is_handover = args.task.startswith("handover")
    hand_times = []
    act_sum = 0.0
    act_count = 0
    for _ in range(args.steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs)
            if args.stochastic:
                actions = actions + torch.randn_like(actions) * action_std
            if args.zero_action:
                actions = torch.zeros_like(actions)
            if is_handover:
                before = ~base_env._released
                if before.any():
                    act_sum += actions[before].abs().mean().item() * int(before.sum())
                    act_count += int(before.sum())
            obs, _, _, _ = env.step(actions)

        # 受け渡しまでの時間を記録
        if is_handover:
            just = base_env._released & before
            if just.any():
                hand_times += (base_env.episode_length_buf[just].float() * base_env.step_dt).tolist()

        # 成功と落下の回数を集計
        num_success += int(base_env._success.sum().item())
        num_drop += int(base_env._dropped.sum().item())
        print(f"\rsuccess {num_success}  drop {num_drop}", end="")
        if diag is not None:
            diag.update()

    print()
    if diag is not None:
        diag.report()

    # エピソードごとの結果を表示
    if is_handover:
        count = min(base_env._outcome_count, base_env.cfg.outcome_window)
        rates = base_env._outcomes[:count].mean(dim=0).tolist() if count > 0 else [0.0] * 5
        mean_t = sum(hand_times) / len(hand_times) if hand_times else 0.0
        mode = "zero" if args.zero_action else "policy"
        print("========== RESULT ==========", flush=True)
        print(f"RESULT {args.task} {mode}: エピソード {count}、受け渡し {rates[1] * 100:.1f} %、"
              f"成功 {rates[0] * 100:.1f} %、落下 {rates[2] * 100:.1f} %、"
              f"受け渡しまで {mean_t:.2f} s（{len(hand_times)} 回）、"
              f"受け渡し前の行動の大きさ |a| {act_sum / max(1, act_count):.3f}（補正 {act_sum / max(1, act_count) * base_env.cfg.residual_limit:.3f} rad）", flush=True)
    env.close()


class HandoverDiag:
    # 受け渡し前の把持判定を集計

    def __init__(self, env):
        self.env = env
        self.keys = ["距離 < 2 cm", "爪の間", "向き", "閉", "爪の間 & 向き", "把持"]
        self.counts = torch.zeros(len(self.keys))
        self.steps = 0
        self.offsets = []
        self.recv_grip = []
        self.giver_grip = []
        self.grip_vel = []
        self.run = torch.zeros(env.num_envs, device=env.device)
        self.max_run = 0

    def update(self):
        e = self.env
        before = ~e._released
        if before.sum() == 0:
            return
        c1 = e._in_jaws
        c2 = c1 & e._closed
        flags = [e._dist < 0.02, e._inside, e._aligned, e._closed, c1, c2]

        # 把持の最長継続を記録
        self.run = torch.where(e._squeezing & before, self.run + 1, torch.zeros_like(self.run))
        self.max_run = max(self.max_run, int(self.run.max().item()))

        # 把持中のグリッパ速度を記録
        grip_id = e._recv_joint_ids[5]
        if c2.any():
            self.grip_vel.append(e.receiver.data.joint_vel[c2, grip_id].abs().mean().item())
        self.counts += torch.tensor([(f & before).sum().item() for f in flags], dtype=torch.float)
        self.steps += int(before.sum().item())

        # 近接時の把持点ずれを記録
        near = before & (e._dist < 0.03)
        if near.any():
            self.offsets.append(e._grasp_in_tcp[near].abs().mean(dim=0).cpu())

        # 押し込み時のグリッパ角を記録
        pressing = before & e._in_jaws & e._pushing
        if pressing.any():
            self.recv_grip.append(e.receiver.data.joint_pos[pressing, grip_id].mean().item())

        # 渡す側のグリッパ角を記録
        self.giver_grip.append(e.giver.data.joint_pos[before, e._giver_joint_ids[5]].mean().item())

    def report(self):
        print("========== 把持判定の診断 ==========")
        for key, c in zip(self.keys, self.counts.tolist()):
            print(f"  {key:16s} {100.0 * c / max(1, self.steps):6.2f} %")
        if self.offsets:
            o = torch.stack(self.offsets).mean(dim=0) * 1000
            w = [v * 1000 for v in self.env.cfg.grasp_window]
            print(f"  把持点のずれ X {o[0]:.1f} / Y {o[1]:.1f} / Z {o[2]:.1f} mm"
                  f" (範囲 {w[0]:.0f} / {w[1]:.0f} / {w[2]:.0f} mm)")
        if self.recv_grip:
            print(f"  押し込み時のグリッパ角 {sum(self.recv_grip) / len(self.recv_grip):.3f} rad"
                  f" (上限 {self.env.cfg.gripper_closed_max} rad)")
        if self.grip_vel:
            print(f"  把持中のグリッパ速度 {sum(self.grip_vel) / len(self.grip_vel):.3f} rad/s"
                  f" (上限 {self.env.cfg.gripper_stall_vel} rad/s)")
        print(f"  把持の最長継続 {self.max_run} step (必要 {self.env.cfg.release_steps})")
        if self.giver_grip:
            print(f"  渡す側のグリッパ角 {sum(self.giver_grip) / len(self.giver_grip):.3f} rad"
                  f" (目標 {self.env.cfg.giver_gripper_closed} rad)")


class PickDiag:
    # 供給側の各条件を集計

    def __init__(self, env):
        self.env = env
        self.before_keys = ["距離 < 2 cm", "爪の間", "爪幅の向き", "手先が下向き", "爪の間 & 向き & 下向き", "把持",
                            "爪の食い込み", "  固定側", "  可動側"]
        self.after_keys = ["受け渡し位置 < 2 cm", "ブロック鉛直", "手先が組立側", "掴んだ端が下", "すべて"]
        self.before = torch.zeros(len(self.before_keys))
        self.after = torch.zeros(len(self.after_keys))
        self.n_before = 0
        self.n_after = 0
        self.grasps = 0
        self.present_dist = []
        self.block_height = []
        self.max_height = 0.0
        self.snap = []

    def update(self):
        e = self.env
        cfg = e.cfg
        b = ~e._attached
        a = e._attached
        if b.any():
            c1 = e._in_jaws
            c2 = c1 & e._closed
            flags = [e._dist < 0.02, e._inside, e._align > cfg.grasp_align_cos, e._down > cfg.grasp_down_cos,
                     c1, c2, e._penetrating, e._pen_hand, e._pen_jaw]
            self.before += torch.tensor([(f & b).sum().item() for f in flags], dtype=torch.float)
            self.n_before += int(b.sum().item())
        if a.any():
            flags = [e._present_dist < cfg.present_tol, e._vertical > cfg.present_vertical_cos,
                     e._facing > cfg.present_approach_cos, e._lower_end > 0.5, e._presenting]
            self.after += torch.tensor([(f & a).sum().item() for f in flags], dtype=torch.float)
            self.n_after += int(a.sum().item())
            self.present_dist.append(e._present_dist[a].mean().item())
            z = e._block_pos[a, 2] - e.scene.env_origins[a, 2]
            self.block_height.append(z.mean().item())
            self.max_height = max(self.max_height, z.max().item())
        self.grasps += int(e._just_grasped.sum().item())
        if e._just_grasped.any():
            self.snap.extend(e._snap_dist[e._just_grasped].tolist())

    def report(self):
        print("========== 供給側の診断 ==========")
        print(f"  把持回数 {self.grasps}")
        print(f"  把持前 ({self.n_before} step)")
        for key, c in zip(self.before_keys, self.before.tolist()):
            print(f"    {key:22s} {100.0 * c / max(1, self.n_before):6.2f} %")
        print(f"  把持後 ({self.n_after} step)")
        for key, c in zip(self.after_keys, self.after.tolist()):
            print(f"    {key:22s} {100.0 * c / max(1, self.n_after):6.2f} %")
        if self.present_dist:
            mean = sum(self.present_dist) / len(self.present_dist)
            print(f"  受け渡し位置との距離 平均 {mean * 1000:.1f} mm")
            h = sum(self.block_height) / len(self.block_height)
            print(f"  ブロック高さ 平均 {h * 1000:.1f} mm / 最大 {self.max_height * 1000:.1f} mm")
        if self.snap:
            print(f"  把持時のブロックの移動量 平均 {sum(self.snap) / len(self.snap) * 1000:.1f} mm / 最大 {max(self.snap) * 1000:.1f} mm")
        e = self.env
        print(f"  範囲外での終了回数 {e.out_events}")
        if e.out_after_grasp:
            mean = sum(e.out_after_grasp) / len(e.out_after_grasp)
            print(f"    把持から範囲外まで 平均 {mean:.2f} s")
            for pos in e.out_positions[:5]:
                print(f"    範囲外時のブロック位置 ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")


if __name__ == "__main__":
    main()
    simulation_app.close()
