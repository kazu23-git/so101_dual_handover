from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 学習の回し方を設定
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 25
    experiment_name = "so101_handover"
    empirical_normalization = False

    # 観測グループを設定（rsl-rl 3.x）
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    # 方策と価値関数のネットワークを設定
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # PPO を設定
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickPPORunnerCfg(HandoverPPORunnerCfg):
    # 学習の回し方を設定
    num_steps_per_env = 32
    save_interval = 25
    experiment_name = "so101_pick_present"

    # 方策と価値関数のネットワークを設定
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # PPO を設定
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PickLowPPORunnerCfg(PickPPORunnerCfg):
    # 低い位置に差し出す学習の設定
    experiment_name = "so101_pick_low"


@configclass
class HandoverTopPPORunnerCfg(HandoverPPORunnerCfg):
    # 上端を真上からつまむ学習の設定
    experiment_name = "so101_handover_top"
