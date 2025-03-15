import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from replay_memory import Replay_Memory

from random import randint
LR = 1e-4
GREEDY_EPSILON = 0.9
BATCH_SIZE = 64
GAMMA = 0.99

OUTPUT_DIMENSION = 80

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(11, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 25 * 25, 120)  # Adjusted input size
        self.fc2 = nn.Linear(120, OUTPUT_DIMENSION)

    def forward(self, x):
        x.to(self.device)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 25 * 25)  # Adjusted flatten size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def to(self, device):
        self.device = device
        self.conv1.to(device)
        self.conv2.to(device)
        self.pool.to(device)
        self.fc1.to(device)
        self.fc2.to(device)

class DQN():
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Network(), Network()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()
        self.isEvalMode = False
        self.memory = Replay_Memory()

    def choose_action(self, encoded_state):
        if self.isEvalMode and np.random.random() <= GREEDY_EPSILON:
            action_values = self.eval_net.forward(encoded_state)
            best_action = torch.max(action_values, 1)[1].data.numpy()
            return best_action
        else:
            return randint(0, OUTPUT_DIMENSION)
    
    def eval(self):
        self.isEvalMode = True

    def store_action(self, state, action, reward, next_state):
        self.memory.store((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.get_sample(BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch)
        next_state_batch = torch.cat(next_state_batch)

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_function(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def to(self, device):
        self.eval_net.to(device)
        self.target_net.to(device)