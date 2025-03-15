
import octospace
import pygame


import torch.nn as nn
from torch.optim import Adam
import torch
import gymnasium as gym
import numpy as np
import os
from my_agent import Agent

import octospace
import pygame


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005



def load_player_for_training(side: int):
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"

    agent = Agent(side)
    try:
        agent.load(os.path.abspath(f"agents/model"))
    except FileNotFoundError:
        print("did not load model from memory!!!")
        pass
    
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

            next_state, reward, terminated, truncated, _ = env.step(
            {
                "player_1": action_1,
                "player_2": action_2
            })
            isFinished = truncated or terminated

            player_left.store(state, action_1,next_state,  reward["player_1"])
            player_right.store(state, action_2, next_state, reward["player_2"])

            if isFinished:
                player_left.learn()
                player_right.learn()
            
            t+=1
            state = next_state 




if __name__ == "__main__":
    main()