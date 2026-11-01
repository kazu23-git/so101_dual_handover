"""学習タスクの一覧。"""

from . import PICK_LOW_TASK_NAME, PICK_TASK_NAME, TASK_NAME, TOP_TASK_NAME
from .agents.rsl_rl_ppo_cfg import (
    HandoverPPORunnerCfg,
    HandoverTopPPORunnerCfg,
    PickLowPPORunnerCfg,
    PickPPORunnerCfg,
)
from .handover_env_cfg import HandoverEnvCfg, HandoverTopEnvCfg
from .pick_env_cfg import PickEnvCfg, PickLowEnvCfg

# タスク名ごとの環境 ID と設定を定義
TASKS = {
    "handover": (TASK_NAME, HandoverEnvCfg, HandoverPPORunnerCfg),
    "pick": (PICK_TASK_NAME, PickEnvCfg, PickPPORunnerCfg),
    "pick_low": (PICK_LOW_TASK_NAME, PickLowEnvCfg, PickLowPPORunnerCfg),
    "handover_top": (TOP_TASK_NAME, HandoverTopEnvCfg, HandoverTopPPORunnerCfg),
}
