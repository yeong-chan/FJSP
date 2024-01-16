import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment.FJSP_env import FJSP

writer = SummaryWriter()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# seed = 12123
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)

if __name__ == "__main__":
    log_path = 'result\\model\\test'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    action_mode = 'random'
    event_path = f'environment\\result\\{action_mode}'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    # env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=action_mode,
    #            log_dir=event_path)
    env = FJSP(num_m=30, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=100, action_mode=action_mode,
               log_dir=event_path)
    print(f"action mode : <{action_mode}>")
    print("ep\treward\tTotal tardiness\t실행시간(s)")
    total_reward = 0
    for e in range(1, 100 + 1):
        seed = 12123 * e
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env.reset()
        start = time.time()
        done = False

        r = list()
        loss = 0
        num_update = 0

        while not done:
            action = 0
            next_state, reward, done = env.step(action)
            r.append(reward)
            state = next_state

        # print("ep:", e, "reward:", np.sum(r), "실행시간:", time.time() - start)
        print(f"{e}\t{np.sum(r):.2f}\t{env.monitor.tardiness:.2f}", "\t", time.time() - start)
        total_reward += np.sum(r)
        if np.sum(r).round() != env.monitor.tardiness.round():
            print(f"@{np.sum(r):.1f}!={env.monitor.tardiness:.1f}")
    print("average reward: ", total_reward / 100)
