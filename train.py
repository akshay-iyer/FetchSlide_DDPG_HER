import torch
import gym
import os
from algorithms.vanilla_ddpg import *
from algorithms.ddpg_with_her import *
from algorithms.ddpg_her_normalizn import *

def train_agent(args):

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    env = gym.make(args.env_name)
    observation = env.reset()

    #print("Initial observation: ", observation)

    env_params = {

        # for hopper v2
        #'obs_dim' : observation.shape[0], #11
        # for fetch slide
        'obs_dim' : observation['observation'].shape[0], #(25,)
        'goal_dim': observation['desired_goal'].shape[0],  #(3,)
        'action_dim': env.action_space.shape[0], #(4,)
        'max_action' : env.action_space.high[0], # high : [1,1,1,1] low: [-1,-1,-1,-1]
    }

    if args.her:
        ddpg_agent = DDPG_HER_N(args, env, env_params)
    else:
        ddpg_agent = DDPG(args, env, env_params)

    ddpg_agent.train()