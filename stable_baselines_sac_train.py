import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from marathon_envs.envs import MarathonEnvs

ENV_NAME = "Walker2d-v0"
ENV_PATH = os.path.join("builds", "Walker2d-v0", "Marathon Environments.exe")
RUN_ID = "sac_walker_01"
TOTAL_TIMESTEPS = 2_000_000
SAVE_FREQ = 100_000


class BestModelCallback(BaseCallback):
    """Saves the model whenever mean episode reward improves."""

    def __init__(self, save_path: str, min_episodes: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.min_episodes = min_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) >= self.min_episodes:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                path = os.path.join(self.save_path, "best_model")
                self.model.save(path)
                if self.verbose:
                    print(f"  → New best: {mean_reward:.1f} (saved {path})")
        return True


save_dir = os.path.join("results", RUN_ID)
os.makedirs(save_dir, exist_ok=True)

env = Monitor(
    MarathonEnvs(
        ENV_NAME,
        num_spawn_envs=1,
        marathon_envs_path=ENV_PATH,
        no_graphics=True,
        multiagent=False,
    ),
    filename=os.path.join(save_dir, "monitor.csv"),
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="summaries",
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1_000_000,
    learning_starts=10_000,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)

callbacks = CallbackList([
    CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=save_dir,
        name_prefix=ENV_NAME,
    ),
    BestModelCallback(save_path=save_dir),
])

start = timer()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, tb_log_name=RUN_ID)
elapsed = timer() - start
print(f"Training time: {timedelta(seconds=elapsed)}")

model.save(os.path.join(save_dir, ENV_NAME + "_final"))
env.close()
