import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from marathon_envs.envs import MarathonEnvs

ENV_NAME = "Walker2d-v0"
ENV_PATH = os.path.join("builds", "Walker2d-v0", "Marathon Environments.exe")
RUN_ID = "sac_walker_02"
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 100_000
N_ENVS = 4


def make_env(worker_id: int):
    def _init():
        return MarathonEnvs(
            ENV_NAME,
            worker_id=worker_id,
            marathon_envs_path=ENV_PATH,
            no_graphics=True,
            multiagent=False,
        )
    return _init


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


class CheckpointWithBufferCallback(BaseCallback):
    """Saves model + replay buffer together at each checkpoint interval."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(self.save_path)
            self.model.save_replay_buffer(self.save_path + "_replay_buffer")
            if self.verbose:
                print(f"  [checkpoint] saved at step {self.num_timesteps}")
            self._last_save = self.num_timesteps
        return True


if __name__ == "__main__":
    save_dir = os.path.join("results", RUN_ID)
    os.makedirs(save_dir, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecMonitor(vec_env, filename=os.path.join(save_dir, "monitor.csv"))

    checkpoint_path = os.path.join(save_dir, ENV_NAME + "_checkpoint")
    replay_buffer_path = checkpoint_path + "_replay_buffer"

    resuming = os.path.exists(checkpoint_path + ".zip")

    if resuming:
        print(f"Resuming from {checkpoint_path}.zip")
        model = SAC.load(checkpoint_path, env=env, tensorboard_log="summaries")
        if os.path.exists(replay_buffer_path + ".pkl"):
            print(f"Loading replay buffer from {replay_buffer_path}.pkl")
            model.load_replay_buffer(replay_buffer_path)
        remaining = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"Resuming at step {model.num_timesteps}, {remaining} steps remaining")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="summaries",
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
            train_freq=4,
            gradient_steps=4,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        remaining = TOTAL_TIMESTEPS

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=save_dir,
            name_prefix=ENV_NAME,
        ),
        CheckpointWithBufferCallback(
            save_freq=SAVE_FREQ,
            save_path=checkpoint_path,
        ),
        BestModelCallback(save_path=save_dir),
    ])

    start = timer()
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        tb_log_name=RUN_ID,
        reset_num_timesteps=not resuming,
    )
    elapsed = timer() - start
    print(f"Training time: {timedelta(seconds=elapsed)}")

    model.save(os.path.join(save_dir, ENV_NAME + "_final"))
    model.save_replay_buffer(checkpoint_path + "_replay_buffer")
    env.close()
