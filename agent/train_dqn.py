import os
import random

import numpy as np
import torch

from agent.dqn import *
from environment.FJSP_env import FJSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    num_episode = 10000
    episode = 1

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)
    mode = 'heuristic'
    state_size = 7
    if mode == 'heuristic':
        action_size = 6
    else:
        action_size = 1
    env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=mode, log_dir=event_path)
    q = Qnet(state_size, action_size).to(device)
    q_target = Qnet(state_size, action_size).to(device)
    optimizer = optim.Adam(q.parameters(), lr=1e-5, eps=1e-06)  # learning rate 변경
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    update_interval = 20
    save_interval = 100
    score = 0
    step = 0
    moving_average = list()
    cumulative_rewards = list()


    for e in range(episode, episode + num_episode + 1):
        import time

        start = time.time()
        done = False
        step = 0
        state = env.reset()
        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = max(0.01, 0.1 - 0.01 * (e / 200))
            step += 1
            action = q.sample_action(torch.from_numpy(state).float().to(device), epsilon)
            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            memory.put((state, action, reward, next_state, done))

            if memory.size() > 2000:
                loss += train(q, q_target, memory, optimizer)
                num_update += 1

            state = next_state
            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())
            if done:
                # writer.add_scalar("episode_reward/train", np.sum(r), e)
                if e % save_interval == 0 and e > 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))
                    print('save model...')
                break

        cumulative_rewards.append(np.sum(r))
        print(e, np.sum(r), time.time() - start)



