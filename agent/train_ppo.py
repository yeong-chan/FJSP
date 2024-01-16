import os
import vessl
import pandas as pd
import torch
from environment.FJSP_env import FJSP
from cfg import get_cfg
from ppo import *
import random

def evaluation(agent):
    np.random.seed(30)
    random.seed(30)
    env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=mode, log_dir=event_path)
    possible_actions = [True] * 8
    eval_score = list()
    for e in range(0, 2):
        s = env.reset()
        update_step = 0
        r_epi = 0.0
        done = False
        while not done:
            a, prob, mask = agent.get_action(s, possible_actions)
            s_prime, r, done = env.step(a)
            r_epi += r
            if done:
                break
            update_step += 1
        eval_score.append(r_epi)
    np.random.seed()
    random.seed()
    return np.mean(eval_score)

if __name__ == "__main__":
    cfg = get_cfg()
    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    mode = 'heuristic'
    if mode == 'heuristic':
        action_size = 8
    else:
        action_size = 1
    event_path = '..\\environment\\result\\ddqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    state_size = 7
    rewards_list = list()
    update_interaval = 1

    agent = Agent(state_size, action_size, lr, gamma, lmbda, eps_clip, K_epoch,update_interaval)
    env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=mode, log_dir=event_path)
    k = 1
    possible_actions = [True] * action_size

    empty_list = []
    for e in range(k, 10000):
        s = env.reset()
        update_step = 0
        r_epi = 0.0
        avg_loss = 0.0
        done = False
        while not done:
            a, prob, mask = agent.get_action(s, possible_actions)
            s_prime, r, done = env.step(a)
            agent.put_data((s, a, r, s_prime, prob[a].item(), mask, done), done)
            r_epi += r
            if done:
                break
            update_step += 1
        print(e, r_epi)
        if e % update_interaval == 0:
            print("훈련시작")

            agent.train(e)

        if e % 10 == 0:

            eval_score = evaluation(agent)
            empty_list.append(eval_score)
            print("평가점수", eval_score)
            env = FJSP(num_m=10, num_job_init=20, num_job_add=50, DDT=1.0, IAT_ave=50, action_mode=mode,
                       log_dir=event_path)

        rewards_list.append(r_epi)
        df = pd.DataFrame(empty_list)
        df.to_csv("result.csv")