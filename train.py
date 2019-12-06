import torch
import gym
import os
from algorithms.ddpg_hopper import *

def train_agent(args):

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    env = gym.make(args.env_name).env
    observation = env.reset()

    print("Initial observation: ", observation)

    env_params = {
        # for fetch slide
        #'obs_dim' : observation['observation'].shape[0], #(25,)

        # for hopper v2
        'obs_dim' : observation.shape[0], #11
        #'goal_dim': observation['desired_goal'].shape[0],  #(3,)
        'action_dim': env.action_space.shape[0], #(4,)
        'max_action' : env.action_space.high[0], # high : [1,1,1,1] low: [-1,-1,-1,-1]
    }

    ddpg_agent = DDPG(args, env, env_params)
    ddpg_agent.train()