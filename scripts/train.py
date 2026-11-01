"""PPO による方策の学習"""

import argparse

from isaaclab.app import AppLauncher

# 引数を設定
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["handover", "pick", "pick_low", "handover_top"], default="handover", help="学習する環境")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None, help="再開するチェックポイント")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim を起動
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from datetime import datetime
from importlib import metadata

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from so101_handover.tasks import TASKS


def main():
    # 設定を作成
    task_id, env_cfg_cls, agent_cfg_cls = TASKS[args.task]
    env_cfg = env_cfg_cls()
    agent_cfg = agent_cfg_cls()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    # rsl-rl の版に合わせて設定を変換
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except ImportError:
        pass

    # ログの保存先を作成
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[log] {log_dir}")

    # 環境を作成
    env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 学習器を作成
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume is not None:
        runner.load(args.resume)

    # 設定を保存
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 学習を実行
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # チェックポイントと次のコマンドを表示
    checkpoint = latest_checkpoint(log_dir)
    rel = os.path.relpath(checkpoint)
    print(f"[checkpoint] {rel}")
    print(f"[play] python scripts/play.py --task {args.task} --checkpoint {rel} --num_envs 16 --diagnose")
    print(f"[resume] python scripts/train.py --task {args.task} --headless --num_envs {env_cfg.scene.num_envs} "
          f"--max_iterations 300 --resume {rel}")

    env.close()


def latest_checkpoint(log_dir: str) -> str:
    # 最新の model_*.pt を取得
    names = [n for n in os.listdir(log_dir) if n.startswith("model_") and n.endswith(".pt")]
    names.sort(key=lambda n: int(n[len("model_"):-len(".pt")]))
    return os.path.join(log_dir, names[-1])


if __name__ == "__main__":
    main()
    simulation_app.close()
