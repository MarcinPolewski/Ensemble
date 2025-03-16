import torch.nn as nn
import encoder

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.epsilon = 0.1

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 128 * 2 * 2)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def get_best_action_pre_one_ship(self, model_output):
        return model_output.argmax().item()
    
    def get_action_pre_one_ship(self, model_output): # FIX THIS METHOD
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return model_output.argmax().item()
    
    def get_all_best_actions(self, obs):
        ship_actions = []
        for ship in obs["allied_ships"]:
            encoded = encoder.encode(obs, ship) 
            model_output = self.model(encoded)
            action_for_ship = self.get_best_action_pre_one_ship(model_output)
            ship_actions.append(action_for_ship)
        return ship_actions
    
    def get_action(self, obs):
        ship_actions = []
        print(obs)
        for ship in obs["allied_ships"]:
            encoded = encoder.encode(obs, ship) 
            model_output = self.model(encoded)
            action_for_ship = self.get_action_pre_one_ship(model_output)
            ship_actions.append(action_for_ship)
        return ship_actions
