from processing_model import CNN
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import octospace
import pygame
import gymnasium as gym

import random

class ModelTrainer:
    def __init__(self):

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"

        # hyper params
        self.psilon = 1e-2
        self.episodes = 1000
        self.lr = 1e-3
        self.target_update_freq = 10
        self.gamma = 0.99

        # Q-networks
        self.batch_size = 128

        self.loss_f = nn.MSELoss()
        self.memory_0 = []
        self.memory_1 = []

        # Players
        self.target_network_0 = CNN()
        self.target_network_0.to(self.device)
        self.policy_network_0 = CNN()
        self.policy_network_0.to(self.device)

        self.target_network_1 = CNN()
        self.target_network_1.to(self.device)
        self.policy_network_1 = CNN()
        self.policy_network_1.to(self.device)


        self.optimizer_0 = Adam(self.policy_network_0.parameters(), lr=self.lr)
        self.optimizer_1 = Adam(self.policy_network_1.parameters(), lr=self.lr)

    def optimize(self, memory, optimizer):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        q_values = (self.policy_net(states) * actions).sum(dim=1)

        with torch.no_grad():
            max_target_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_target_q_values * (1 - dones)

        loss = self.loss_f(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000, turn_on_music=False, volume=0.1)

        for episode in range(200):
            state, _ = env.reset()
            done = False

            episode_reward = {
                "player_1": 0,
                "player_2": 0
            }
            steps = 0
            while not done:
                #state = torch.FloatTensor(state[0])
                action_0 = self.policy_network_0.get_action(state["player_1"])  
                action_1 = self.policy_network_1.get_action(state["player_2"])

                action = {action_0, action_1}



                next_state, reward, done, _, _ = env.step(action)

                # memory
                self.memory_0.append((state["player_1"]), action_0, float(reward["player_0"]))
                self.memory_1.append((state["player_1"]), action_1, float(reward["player_1"]))

                # update state
                state = next_state
                episode_reward["player_1"] += reward["player_1"]
                episode_reward["player_2"] += reward["player_2"]

                # optimize
                self.optimize(self.memory_0, self.optimizer_0)
                self.optimize(self.memory_1, self.optimizer_1)

                if steps % self.target_update_freq == 0:
                    self.target_network_0.load_state_dict(self.policy_network_0.state_dict())
                    self.target_network_1.load_state_dict(self.policy_network_1.state_dict())
                
                steps += 1

        self.save(self.target_network_0, "model_0")
        self.save(self.target_network_1, "model_1")


    def save(self, model, model_name):
        torch.save(model.state_dict(), model_name)

    def load(self, abs_path: str):
        self.agent.load_state_dict(torch.load(abs_path))

    def eval(self):
        self.agent.eval()

    def to(self, device):
        self.agent.to(device)

def main():
    m = ModelTrainer()
    m.train()

if __name__ == "__main__":
    main()