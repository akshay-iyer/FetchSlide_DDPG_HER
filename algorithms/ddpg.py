import torch
import os
from mpi4py import MPI
import numpy as np

from networks.actor_critic import *

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.idx = 0
        self.size = 0
        self.max_size = size

        self.obs1_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.reward_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.done_buffer   = np.zeros([size, obs_dim], dtype=np.float32)

    def store(self, obs, next_obs, action, reward, done):
        self.obs1_buffer[self.idx]   = obs
        self.obs2_buffer[self.idx]   = next_obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.done_buffer[self.idx]   = done

        self.idx  = (self.idx+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        random_idxs = np.random.randint(0,self.size, batch_size)

        return dict(
            obs1   = self.obs1_buffer[random_idxs],
            obs2   = self.obs2_buffer[random_idxs],
            action = self.action_buffer[random_idxs],
            reward = self.reward_buffer[random_idxs],
            done   = self.done_buffer[random_idxs],
        )



class DDPG:
    def __init__(self, args, env, env_params):
        # actor = policy network
        # critic = Q network

        # create actor critic pair
        self.actor  = Actor(env_params)
        self.critic = Critic(env_params)

        # create target networks which lag the original networks
        self.actor_target = Actor(args.env_params)
        self.critic_target = Critic(args.env_params)

        # loading main params into target for the first time
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # using Adam optimizer
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        self.env = env
        self.env_params = env_params
        self.args = args

        self.buffer = ReplayBuffer(self.env_params['obs_dim'], self.env_params['action_dim'], self.args.buffer_size)

    def generate_action_with_noise(self, obs, noise):
        action = self.actor(torch.Tensor(obs.reshape(1,-1)))
        action = action.cpu().numpy().squeeze() + noise*np.random.randn(self.env_params['action_dim'])
        return action


    def train(self):

        total_steps    = self.args.epochs*self.args.steps_in_epoch
        episode_reward = 0
        episode_len    = 0
        done           = False
        obs            = self.env.reset()
        o = obs['observation']

        for step in range(total_steps):
            self.actor.eval()
            self.critic.eval()

            # initial random exploration
            if(step < self.args.start_steps):
                a = self.env.action_space.sample()
            else:
                a = self.generate_action_with_noise(o, self.args.noise_scale)

            # take one step
            o2, r, d, _ = self.env.step(a)
            o2 = o2['observation']
            episode_reward += r
            episode_len +=1

            d = False if episode_len == self.args.max_ep_len else d

            # store experience in buffer
            self.buffer.store(o, o2, a, r, d)

            # update observation
            o=o2






