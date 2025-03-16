# Skeleton for Agent class
import torch
import torch.nn as nn
import random
import numpy as np
import os

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
        x = x.to(self.device)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        return x

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


def select_best_action(state, device, model):
        allies_indexes = [state["allied_ships"][i][0] for i in range(len(state["allied_ships"]))]
        encoded_state = torch.FloatTensor(encode_obs(state)).to(device)
        q_values = model(encoded_state)
        return decode_action(torch.argmax(q_values.view(-1, 8), dim=1), allies_indexes)


class Agent:
    def __init__(self, side: int):
        """
        :param side: Indicates whether the player is on left side (0) or right side (1)
        """
        self.side = side

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

        action = select_best_action(obs, self.device, self.model)
        return action

    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """

        self.model = DQN(151, 80)
        self.model.load_state_dict(torch.load(os.path.join(abs_path, "target_model.pt"), map_location=torch.device('cpu') if not torch.cuda.is_available() else None))
        self.eval()

    def eval(self):
        """
        With this function you should switch the agent to inference mode.

        :return:
        """
        self.model.eval()

    def to(self, device):
        """
        This function allows you to move the agent to a GPU. Please keep that in mind,
        because it can significantly speed up the computations and let you meet the time requirements.

        :param device:
        :return:
        """

        self.model.to(device)
        self.model.device = device
        self.device = device