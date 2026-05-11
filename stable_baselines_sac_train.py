import os
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from marathon_envs.envs import MarathonEnvs

ENV_NAME = "Walker2d-v0"
ENV_PATH = os.path.join("builds", "Walker2d-v0", "Marathon Environments.exe")
RUN_ID = "sac_walker_01"
TOTAL_TIMESTEPS = 2_000_000
SAVE_FREQ = 100_000

env = MarathonEnvs(
    ENV_NAME,
    num_spawn_envs=1,
    marathon_envs_path=ENV_PATH,
    no_graphics=True,
    multiagent=False,
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

checkpoint_cb = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=os.path.join("results", RUN_ID),
    name_prefix=ENV_NAME,
)

start = timer()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, tb_log_name=RUN_ID)
elapsed = timer() - start
print(f"Training time: {timedelta(seconds=elapsed)}")

model.save(os.path.join("results", RUN_ID, ENV_NAME + "_final"))
env.close()
