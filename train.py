import torch
import gym
from algorithms.ddpg import *

def train_agent(args):
    env = gym.make(args.env_name)
    observation = env.reset()

    print("Initial observation: ", observation)

    env_params = {
        'obs_dim' : observation['observation'].shape[0], #(25,)
        'goal_dim': observation['desired_goal'].shape[0],  #(3,)
        'action_dim': env.action_space.shape[0], #(4,)
        'max_action' : env.action_space.high[0], # high : [1,1,1,1] low: [-1,-1,-1,-1]
    }

    ddpg_agent = DDPG(args, env, env_params)
    ddpg_agent.train()