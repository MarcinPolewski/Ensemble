
import torch.nn as nn
from torch.optim import Adam
import torch
import gymnasium as gym
import numpy as np
import random
import os

import octospace
import pygame
import my_agent

def load_player_for_training(side: int):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = my_agent(side)
    # agent.load(os.path.abspath(f"agents/player"))
    agent.to(DEVICE)

    return agent

def main(): 
    # create enviroment
    env = gym.make('OctoSpace-v0', player_1_id=1, player_2_id=2, max_steps=1000,
                render_mode="human", turn_on_music=False, volume=0.1)
    state, _ = env.reset()

    player_left = load_player_for_training(0)
    player_right = load_player_for_training(1)

    max_episode_count = 1000
    max_iterations_per_episode = 1000
    terminated = False
    
    for _ in range(max_episode_count):
        
        state,_ = env.reset()
        isFinished = False

        t=0
        while(t < max_iterations_per_episode and not isFinished):
            action_1 = player_left.get_action(state["player_1"]) 
            action_2 = player_right.get_action(state["player_2"]) 

            next_state, reward, terminated, truncated, info = env.step(
            {
                "player_1": action_1,
                "player_2": action_2
            })
            isFinished = truncated or terminated

            player_left.store(state, action_1, reward["player_1"], next_state)
            player_right.store(state, action_2, reward["player_2"], next_state)

            if isFinished:
                player_left.learn()
                player_right.learn()
            
            t+=1
            state = next_state 




if __name__ == "__main__":
    main()