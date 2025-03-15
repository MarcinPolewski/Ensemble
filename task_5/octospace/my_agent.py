import torch
import numpy as np
from replay_memory import Replay_Memory
from random import randint

from DQN import DQN

def encode_observation(observation: dict) -> torch.Tensor:

    tensor_3d = torch.zeros((100, 100, 11), dtype=torch.float32)

    map_tensor = torch.tensor(observation["map"], dtype=torch.float32).to(torch.int)

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
    if predicted_class > 3:
        return [ship_id, 1, predicted_class - 4, 0]
    else:
        return [ship_id, 0, predicted_class, 3]



class Agent:
    def __init__(self, side: int):
        self.memory = Replay_Memory()
        self.side = side
        self.model = DQN()

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

        encoded_obs = encode_observation(obs)
        model_output = self.model.choose_action(encoded_obs)
        ship_actions = self.parse_model_output(obs["allied_ships"],model_output)

        return {
            "ships_actions": ship_actions,
            "construction": 10
        }
    
    def parse_model_output(self, ships, model_output):
        ship_idx = model_output // 8
        move_idx = model_output % 8 
        move_type = 0

        if move_idx >= 4:
            move_type = 1
            move_idx -= 4

        if(ship_idx > len(ships)):
            ship_idx = randint(0, len(ships)-1)

        print(ship_idx)

        return ships[ship_idx], move_type, move_idx

    def store(self, state, action, reward, next_state):
        self.model.store_action(state, action, reward, next_state)

    def load(self, abs_path: str):
        """
        Function for loading all necessary weights for the agent. The abs_path is a path pointing to the directory,
        where the weights for the agent are stored, so remember to join it to any path while loading.

        :param abs_path:
        :return:
        """
        state_dictionary = torch.load(f"{abs_path}_{self.side}")
        self.model = DQN()
        self.model.load_state_dict(state_dictionary)


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
