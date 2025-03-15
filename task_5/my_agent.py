# Skeleton for Agent class


import torch
import torch.nn as nn
import torch.nn.functional as F


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


def decode_action(action: torch.Tensor, ship_id: int) -> list:
    print(f"shape: {action.shape}")
    probabilities = F.softmax(action, dim=0)
    predicted_class = torch.argmax(probabilities)
    predicted_class = predicted_class.item()
    print(predicted_class)
    if predicted_class > 3:
        return [ship_id, 1, predicted_class - 4, 3]
    else:
        return [ship_id, 0, predicted_class, 3]


class Agent:

    def __init__(self, side=0):
        self.device = DEVICE
        self.side = side
        self.leftModel = ConvNet().to(DEVICE)
        self.rightModel = ConvNet().to(DEVICE)

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

            print(encoded_obs.shape)

            if self.side == 0:
                output_tensor = self.leftModel(encoded_obs)
            else:
                output_tensor = self.rightModel(encoded_obs)

            output_tensor = output_tensor.squeeze(1)

            # TODO: decode output from the DQN and add the correct action to the list
            action = decode_action(output_tensor, ship_id)
            ship_actions.append(action)

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
        self.leftModel.to(device)
        self.rightModel.to(device)
