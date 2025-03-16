from agent import Agent
from processing_model import CNN
import torch
import torch.nn as nn
from torch.optim import Adam
from numpy import np
import gym
import random

class ModelTrainer:
    def __init__(self, device):
        self.env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000, turn_on_music=False, volume=0.1)

        # hyper params
        self.psilon = 1e-2
        self.episodes = 1000
        self.lr = 1e-3
        self.target_update_freq = 100000
        self.gamma = 0.99

        # Q-networks
        self.batch_size = 128
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_f = nn.MSELoss()
        self.memory = []

        # Players
        self.player_0 = Agent(0)
        self.player_0.policy_network = CNN()
        self.player_1 = Agent(1)
        self.player_1.policy_network = CNN()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
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

    def train(self, env, num_episodes):
        env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000, turn_on_music=False, volume=0.1)

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False

            while not done:
                state = torch.FloatTensor(state[0])
                action = self.agent.get_action(state)


                self.memory.append((state, action, reward, next_state[0], done))
                self.optimize()

                state = next_state[0]

                if terminated:
                    print(f"Episode {episode} finished after {t+1} timesteps")
                    break

            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

         
                total_reward += reward
                state = next_state

            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns

            log_probs = torch.stack(log_probs)
            values = torch.stack(values)

            advantage = returns - values.detach()

            policy_loss = -(log_probs * advantage).mean()
            value_loss = self.loss_function(values, returns)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def save(self, abs_path: str):
        torch.save(self.agent.state_dict(), abs_path)

    def load(self, abs_path: str):
        self.agent.load_state_dict(torch.load(abs_path))

    def eval(self):
        self.agent.eval()

    def to(self, device):
        self.agent.to(device)

def main():
    # create enviroment
    env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000,
                render_mode="human", turn_on_music=False, volume=0.1)
    state, _ = env.reset()

    agent_1 = Agent(0)

    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()
    gamma = 0.99

    trainer = ModelTrainer(agent, optimizer, loss_function, gamma)
    trainer.to("cuda")

    trainer.train(env, num_episodes=1000)
    trainer.save("model.pth")