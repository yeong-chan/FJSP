import os
import random

import numpy as np
import torch

from agent.ddqn import DDQNAgent
from environment.FJSP_env import FJSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    mode = 'heuristic'

    if mode == 'heuristic':
        action_size = 6
    else:
        action_size = 1

    # state_size = 104  # feature 1~8 size = 104 / 176
    log_path = 'result\\ddqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '..\\environment\\result\\ddqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=mode, log_dir=event_path)

    seed = 12123


    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    memory_size = 10000
    replay_buffer_size = 1000  # N
    batch_size = 32
    learning_rate = 1e-4  # Not given
    gamma = 0.9
    tau = 1e-2  # soft update strategy

    # train
    agent = DDQNAgent(env, state_size=7, action_size=action_size, learning_rate=learning_rate,
                      replay_buffer_size=replay_buffer_size, gamma=gamma, tau=1e-2, update_every=10,
                      batch_size=batch_size)

    training_episodes = 1000

    agent.train(training_episodes)
