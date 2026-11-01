import gymnasium as gym

TASK_NAME = "SO101-Dual-Handover-v0"
PICK_TASK_NAME = "SO101-Pick-Present-v0"
PICK_LOW_TASK_NAME = "SO101-Pick-Present-Low-v0"
TOP_TASK_NAME = "SO101-Handover-Top-v0"

# 受け渡し環境を登録
gym.register(
    id=TASK_NAME,
    entry_point=f"{__name__}.handover_env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:HandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HandoverPPORunnerCfg",
    },
)

# 把持・差し出し環境を登録
gym.register(
    id=PICK_TASK_NAME,
    entry_point=f"{__name__}.pick_env:PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_env_cfg:PickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PickPPORunnerCfg",
    },
)

# 低い位置に差し出す環境を登録
gym.register(
    id=PICK_LOW_TASK_NAME,
    entry_point=f"{__name__}.pick_env:PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_env_cfg:PickLowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PickLowPPORunnerCfg",
    },
)

# 上端を真上からつまむ受け渡し環境を登録
gym.register(
    id=TOP_TASK_NAME,
    entry_point=f"{__name__}.handover_env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:HandoverTopEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:HandoverTopPPORunnerCfg",
    },
)
