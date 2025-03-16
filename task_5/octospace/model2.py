import torch.nn as nn
from torch.optim import Adam
import torch
import gymnasium as gym
import numpy as np
import random

import octospace
import pygame


if torch.cuda.is_available(): 
    device = "cuda"
else:
    device = "cpu"

# Neural network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 2048)
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.to(device)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        return x


# create enviroment
env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000, turn_on_music=False, volume=0.1)

# hyper params
epsilon = 1e-2
episodes = 1000
lr = 1e-3
target_update_freq = 100000
gamma = 0.99

# Q-networks
input_size = 151
output_size = 80
batch_size = 128
policy_net = DQN(input_size, output_size)
target_net = DQN(input_size, output_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = Adam(policy_net.parameters(), lr=lr)
loss_f = nn.MSELoss()
memory = []


def encode_obs(p_obs):
    ships_mask1 = np.array([0, 0, 0, -1, 0, 0] * 10)
    ships_mask2 = np.array([0.001, 0.01, 0.01, 0.01, 0.1, 1/3] * 10)

    allied_ships = np.array(p_obs["allied_ships"]).flatten()
    if len(allied_ships) < 60:
        allied_ships = np.pad(allied_ships, (0, 60 - len(allied_ships)), 'constant')
    allied_ships = (allied_ships[:60] + ships_mask1) * ships_mask2

    enemy_ships = np.array(p_obs["enemy_ships"]).flatten()
    if len(enemy_ships) < 60:
        enemy_ships = np.pad(enemy_ships, (0, 60 - len(enemy_ships)), 'constant')
    enemy_ships = (enemy_ships[:60] + ships_mask1) * ships_mask2

    planets_mask1 = [0, 0, 1] * 9
    planets_mask2 = [0.01, 0.01, 1/101] * 9

    planets_occupation = np.array(p_obs["planets_occupation"]).flatten()
    planets_occupation = np.pad(planets_occupation, (0, 27 - len(planets_occupation)), 'constant')
    planets_occupation = (planets_occupation[:27] + planets_mask1) * planets_mask2

    resources_mask = [0.001] * 4

    resources = np.array(p_obs["resources"]).flatten()
    resources = resources * resources_mask

    encoded = np.concatenate((allied_ships, enemy_ships, planets_occupation, resources))
    return np.array(encoded)


def encode_action(p_action):
    encoded = np.zeros(80)
    ships_actions = p_action["ships_actions"][:10]
    for idx, actions in enumerate(ships_actions):
        encoded[idx * 8 + 4 * actions[1] + actions[2]] = 1
    return np.array(encoded)


def decode_action(p_action, indexes):
    decoded_action = {
        "ships_actions": [],
        "construction": 1
    }

    actions = [
        (0, 0),     # move right
        (0, 1),     # move down
        (0, 2),     # move left
        (0, 3),     # move up
        (1, 0),     # fire right
        (1, 1),     # fire down
        (1, 2),     # fire left
        (1, 3)      # fire up
    ]

    for idx, action_num in zip(indexes[:10], p_action):
        type, direction = actions[action_num]
        if type == 0:
            decoded_action["ships_actions"].append((idx, type, direction, 2))
        else:
            decoded_action["ships_actions"].append((idx, type, direction))

    return decoded_action

def select_best_action(state):
        allies_indexes = [state["allied_ships"][i][0] for i in range(len(state["allied_ships"]))]
        encoded_state = torch.FloatTensor(encode_obs(state)).to(device)
        q_values = policy_net(encoded_state)
        return decode_action(torch.argmax(q_values.view(-1, 8), dim=1), allies_indexes)

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon):
    allies_indexes = [state["allied_ships"][i][0] for i in range(len(state["allied_ships"]))]
    if random.random() < epsilon:
        return decode_action(np.random.randint(0, 8, 10), allies_indexes)
    else:
        encoded_state = torch.FloatTensor(encode_obs(state)).to(device)
        q_values = policy_net(encoded_state)
        return decode_action(torch.argmax(q_values.view(-1, 8), dim=1), allies_indexes)


# Optimization
def optimize():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

    q_values = (policy_net(states) * actions).sum(dim=1)

    with torch.no_grad():
        max_target_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_target_q_values * (1 - dones)

    loss = loss_f(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# training loop
steps = 0

policy_net.to(device)
target_net.to(device)

for episode in range(episodes):
    state, _ = env.reset()
    action = env.action_space.sample()
    done = False
    episode_reward = {
        "player_1": 0,
        "player_2": 0
    }

    while not done:
        # select action
        p1_action = select_action(state["player_1"], epsilon)
        p2_action = select_action(state["player_2"], epsilon)

        action = {
            "player_1": p1_action,
            "player_2": p2_action
        }

        next_state, reward, done, _, info = env.step(action)

        # memory
        memory.append((encode_obs(state["player_1"]), encode_action(p1_action), float(reward["player_1"]), encode_obs(next_state["player_1"]), float(done)))
        memory.append((encode_obs(state["player_2"]), encode_action(p2_action), float(reward["player_2"]), encode_obs(next_state["player_2"]), float(done)))

        # update state
        state = next_state
        episode_reward["player_1"] += reward["player_1"]
        episode_reward["player_2"] += reward["player_2"]

        # optimize
        optimize()

        # update target network
        if steps % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # update steps
        steps += 1

    print(f"Episode: {episode}, reward: {episode_reward}")

# save model
torch.save(policy_net.state_dict(), "policy_model")
torch.save(target_net.state_dict(), "target_model")