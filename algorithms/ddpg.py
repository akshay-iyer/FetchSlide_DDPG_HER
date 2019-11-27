import torch
import os
from mpi4py import MPI

from networks.actor_critic import *

class DDPG:
    def __init__(self, args):
        # create actor critic pair
        self.actor  = Actor(args.env_params)
        self.critic = Critic(args.env_params)
        # sync_networks(self.actor)
        # sync_networks(self.critic)

        # create target networks which lag the original networks
        self.actor_target = Actor(args.env_params)
        self.criic_target = Critic(args.env_params)

        # loading params into target for the first time
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.criic_target.load_state_dict(self.critic.state_dict())

        # using Adam optimizer
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

