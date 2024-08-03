from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = 'Pendulum-v1'
    # env_name: str = 'InvertedPendulumPyBulletEnv-v0'
    episodes: int = 250

    episode: int = 250
    print_at: int = 5
    avg_over: int = 20

    max_experience: int = 1_000_000
    batch_size: int = 64

    log_std_min: float = -5.
    log_std_max: float = 2.
    gamma: float = 0.99
    tau: float = 0.01

    init_temperature: float = 0.2
    ff1_dim: int = 256
    ff2_dim: int = 256
    w_init: float = 3e-3

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    delay_actor_update: int = 1
    delay_target_critic_update: int = 2

    ckpt_dir: str = "./ckpt"
    log_dir: str = "./logs"