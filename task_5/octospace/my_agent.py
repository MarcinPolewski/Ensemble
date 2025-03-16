# Skeleton for Agent class


import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from collections import namedtuple, deque
import random
import torch.optim as optim

import octospace


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def encode_observation(observation: dict) -> torch.Tensor:

    tensor_3d = torch.zeros((100, 100, 11), dtype=torch.float32).to(DEVICE)

    map_tensor = torch.tensor(observation["map"], dtype=torch.float32).to(DEVICE).to(torch.int)

    tensor_3d[:, :, 0] = torch.where(map_tensor == -1, torch.tensor(0), torch.tensor(1))  # is it visible

    tensor_3d[:, :, 1] = torch.where(torch.bitwise_and(map_tensor, 1) != 0, torch.tensor(1), torch.tensor(0))  # is it a planet
    tensor_3d[:, :, 2] = torch.where(torch.bitwise_and(map_tensor, 2) != 0, torch.tensor(1), torch.tensor(0))  # is it an asteroid
    tensor_3d[:, :, 3] = torch.where(torch.bitwise_and(map_tensor, 4) != 0, torch.tensor(1), torch.tensor(0))  # is it an IMF
    tensor_3d[:, :, 4] = torch.where(torch.bitwise_and(map_tensor, 64) != 0, torch.tensor(1), torch.tensor(0))  # is it mine
    tensor_3d[:, :, 5] = torch.where(torch.bitwise_and(map_tensor, 128) != 0, torch.tensor(1), torch.tensor(0))  # is it the opponents

    for ship in observation["allied_ships"]:
        ship_id, x, y, health, fire_cooldown, move_cooldown = ship
        tensor_3d[y, x, 6] = 1

    for ship in observation["enemy_ships"]:
        ship_id, x, y, health, fire_cooldown, move_cooldown = ship
        tensor_3d[y, x, 7] = 1

    tensor_3d[:, :, 8] = torch.where(map_tensor == -1, torch.tensor(0), torch.tensor(1))  # is the planet occupied

    for planet in observation["planets_occupation"]:
        x, y, occupation = planet
        tensor_3d[y, x, 9] = occupation

    return tensor_3d


def decode_action(action: torch.Tensor, ship_id: int) -> list:
    probabilities = F.softmax(action, dim=0)
    predicted_class = torch.argmax(probabilities)
    predicted_class = predicted_class.item()
    if predicted_class > 3:
        return [ship_id, 1, predicted_class - 4, 3]
    else:
        return [ship_id, 0, predicted_class, 3]


def random_action(ship_id: int) -> list:
    return [ship_id, random.randint(0, 1), random.randint(0, 3), 3]


# DQN

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
EPSILON = 0.1

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(11, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(25, 120)  # Adjusted input size
        self.fc2 = nn.Linear(120, 8)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x.reshape(x.size(0), -1)
        # x = x.view(-1, 32 * 25 * 25)  # Adjusted flatten size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent:

    def __init__(self, side=0):
        self.device = DEVICE
        self.side = side
        self.model = ConvNet().to(DEVICE)
        self.eval_on = False

    def get_action(self, obs: dict) -> dict:
        """
        Main function, which gets called during step() of the environment.

        Observation space:
            game_map: whole grid of board_size, which already has applied visibility mask on it
            allied_ships: an array of all currently available ships for the player. The ships are represented as a list:
                (ship id, position x, y, current health points, firing_cooldown, move_cooldown)
                - ship id: int [0, 1000]
                - position x: int [0, 100]
                - position y: int [0, 100]
                - health points: int [1, 100]
                - firing_cooldown: int [0, 10]
                - move_cooldown: int [0, 3]
            enemy_ships: same, but for the opposing player ships
            planets_occupation: for each visible planet, it shows the occupation progress:
                - planet_x: int [0, 100]
                - planet_y: int [0, 100]
                - occupation_progress: int [-1, 100]:
                    -1: planet is unoccupied
                    0: planet occupied by the 1st player
                    100: planet occupied by the 2nd player
                    Values between indicate an ongoing conflict for the ownership of the planet
            resources: current resources available for building

        Action space:
            ships_actions: player can provide an action to be executed by every of his ships.
                The command looks as follows:
                - ship_id: int [0, 1000]
                - action_type: int [0, 1]
                    0 - move
                    1 - fire
                - direction: int [0, 3] - direction of movement or firing
                    0 - right
                    1 - down
                    2 - left
                    3 - up
                - speed (not applicable when firing): int [0, 3] - a number of fields to move
            construction: int [0, 10] - a number of ships to be constructed

        :param obs:
        :return:
        """

        ship_actions = []

        encoded_obs = encode_observation(obs).unsqueeze(0).to(DEVICE)

        for ship in obs["allied_ships"]:
            ship_id, x, y, health, fire_cooldown, move_cooldown = ship
            encoded_obs[0, y, x, 10] = 1

            if self.side == 0:
                output_tensor = self.model(encoded_obs)
            else:
                output_tensor = self.rightModel(encoded_obs)

            output_tensor = output_tensor.squeeze(1)

            # TODO: decode output from the DQN and add the correct action to the list
            action = decode_action(output_tensor, ship_id)

            ship_actions.append(action)

            if (not self.eval_on and random.random() < EPSILON):
                ship_actions[-1] = random_action(ship_id)

        return {
            "ships_actions": ship_actions,
            "construction": 10
        }

    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """
        pass

    def eval(self):
        """
        With this function you should switch the agent to inference mode.

        :return:
        """
        pass

    def to(self, device):
        """
        This function allows you to move the agent to a GPU. Please keep that in mind,
        because it can significantly speed up the computations and let you meet the time requirements.

        :param device:
        :return:
        """
        self.device = device
        self.model.to(device)


def learn_main():
    env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000)

    policy_net = ConvNet().to(DEVICE)
    target_net = ConvNet().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0


if __name__ == "__main__":
    learn_main()
