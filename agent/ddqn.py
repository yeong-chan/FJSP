import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
gamma = 0.9
learning_rate = 1e-5
batch_size = 32

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# device= torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):  # buffer에서 batch_size 길이의 list sample
        transition = random.sample(self.memory, batch_size)
        return transition

    def __len__(self):  # memory 길이 반환
        return len(self.memory)


# Define the network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):  # activation func : tansig
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, 30)
        self.fc6 = nn.Linear(30, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)
        return x


# Define the Double DQN agent
class DDQNAgent:
    def __init__(self, env, state_size, action_size, learning_rate=learning_rate, replay_buffer_size=1000, gamma=gamma,
                 tau=1e-2, update_every=10, batch_size=32):
        self.env = env  # Simpy environment
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        self.steps = 0

        self.qnetwork_local = QNetwork(state_size, action_size).to(device)  # policy
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)  # target

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        # self.update_target_network()

        self.transition = []  # transition to push(store) in memory
        self.is_test = False

    def select_action(self, state: np.ndarray, num_episode: int) -> int:
        eps_start = 0.5
        eps_end = 0.1
        sample = random.random()
        epsilon = eps_start + (eps_end - eps_start) * (self.steps / num_episode)

        if sample > epsilon:  # 최적의 행동 선택 (exploit)
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):  # state가 torch.tensor가 아닐 경우 변환
                    state = torch.tensor(state, dtype=torch.float32)
                return self.qnetwork_local(state).max(dim=0)[1].item()
        else:  # 무작위 행동 선택 (explore)
            return random.randrange(self.action_size)
        # return random.randrange(self.action_size)

    def step(self, action: int):
        next_state, reward, done = self.env.step(action)
        self.transition += [next_state, reward]
        if not self.is_test:
            # Save experience in replay buffer
            self.replay_buffer.push(*self.transition)
            self.steps += 1

        return next_state, reward, done

    def train(self, num_episode: int, plotting_interval=100):

        self.is_test = False
        losses = []
        scores = []
        for e in range(1, num_episode + 1):  # episode = 1 : L
            seed = 12123 # * e
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.select_action(state, num_episode)
                # print(action)
                self.transition = [state, action]
                next_state, reward, done = self.step(action)
                # print(f"s':{next_state}\ta:{action}\tr:{reward}")
                reward /= 300
                state = next_state
                score += reward
                if done:  # episode done
                    scores.append(score)
                    print(f"e:{e}, score:{score}, Total tardiness: {self.env.monitor.tardiness:.1f}")
                if len(self.replay_buffer) >= self.batch_size:
                    transitions = self.replay_buffer.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))

                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                                  dtype=torch.bool)
                    non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=device).view(1, -1) for s in batch.next_state if s is not None])

                    state_batch = torch.cat(
                        [torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) for s in batch.state])
                    # print("state_batch:", state_batch.size())  # 32*7(s)
                    action_batch = torch.cat(
                        [torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(0) for a in batch.action])
                    # print("action_batch size:", action_batch.size())  # 32*1
                    reward_batch = torch.cat(
                        [torch.tensor(r, dtype=torch.int64, device=device).unsqueeze(0) for r in batch.reward])
                    # print("reward batch size:", reward_batch.size())  # 32*1(r)
                    # Q_expected: the Q values of the current state-action pairs
                    # print("qnetwork_local(state_batch)", self.qnetwork_local(state_batch).size())  # 32*6
                    Q_expected = self.qnetwork_local(state_batch).gather(1, action_batch.unsqueeze(-1))
                    # print("Q_expected size:", Q_expected.size())  # 32*1
                    # Q_targets_next: the maximum Q values for the next states
                    Q_targets_next = torch.zeros(self.batch_size, device=device)

                    next_state_actions = self.qnetwork_local(non_final_next_states).max(dim=1)[1].unsqueeze(1)
                    # print("next_state_actions", next_state_actions)  # 6*1
                    #
                    # print("non_final_mask:", non_final_mask.size())  # 32
                    # print("Q_targets_next[non_final_mask]:", Q_targets_next[non_final_mask].size())  # 32
                    # print("non_final_next_states", non_final_next_states.size())  # 32*7
                    # print(self.qnetwork_target(non_final_next_states).size())  # 32*6
                    Q_targets_next[non_final_mask] = self.qnetwork_target(non_final_next_states).gather(1, next_state_actions).squeeze()

                    # Q_targets: the target Q values
                    Q_targets = reward_batch + (self.gamma * Q_targets_next)  # 32*1 ??
                    # print("Q_targets", Q_targets.size())
                    # print("Q_expected", Q_expected.size())

                    # Compute loss
                    loss = F.mse_loss(Q_expected, Q_targets.unsqueeze(1).detach())

                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.qnetwork_local.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
                    losses.append(loss)

                    # Update target network
                    if e % self.update_every == 0:
                        self.update_target_network()

            if e % plotting_interval == 0:
                self.evaluation(num_episode)

    def evaluation(self, num_episode):
        saved_np_seed = np.random.get_state()
        saved_random_seed = random.getstate()
        saved_torch_state = torch.get_rng_state()

        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        num_ep = 30
        scores = []
        update_cnt = 0
        for e in range(1, num_ep + 1):
            self.is_test = True
            done = False
            state = self.env.reset()
            score = 0
            while not done:
                action = self.select_action(state, num_episode)
                self.transition = [state, action]
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
            scores.append(score)  # print(f"e:{e}, score:{score}")

        print(f"{num_ep}ep 평균 eval score:{np.mean(scores)}")

        np.random.set_state(saved_np_seed)
        random.setstate(saved_random_seed)
        torch.set_rng_state(saved_torch_state)

    def update_target_network(self):
        # Update target network parameters with polyak averaging
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
