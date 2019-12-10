import torch
import torch.nn.functional as F
import gym
import os
import random
import copy
import numpy as np
import sys
from time import localtime, strftime

from networks.actor_critic import *
from .her import *
from .normalizer import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.idx = 0
        self.size = 0
        self.max_size = size

        self.obs1_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buffer = np.zeros([size, act_dim], dtype=np.float32)
        self.reward_buffer = np.zeros(size           , dtype=np.float32)
        self.goal_buffer   = np.zeros([size, goal_dim],dtype=np.float32)
        self.done_buffer   = np.zeros(size           , dtype=np.float32)
        self.type_buffer   = np.zeros(size           , dtype=np.float32)


    def _store(self, obs, next_obs, action, reward, goal, done, replay_type):
        self.obs1_buffer[self.idx]   = obs
        self.obs2_buffer[self.idx]   = next_obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.goal_buffer[self.idx]   = goal
        self.done_buffer[self.idx]   = done
        self.type_buffer[self.idx]   = replay_type


        self.idx  = (self.idx+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

    def _sample_batch(self, batch_size=128):
        random_idxs = np.random.randint(0,self.size, batch_size)

        return dict(
            obs1          = self.obs1_buffer[random_idxs],
            obs2          = self.obs2_buffer[random_idxs],
            action        = self.action_buffer[random_idxs],
            reward        = self.reward_buffer[random_idxs],
            goal          = self.goal_buffer[random_idxs],
            done          = self.done_buffer[random_idxs],
            replay_type   = self.type_buffer[random_idxs],
        )

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class DDPG_HER_N:
    def __init__(self, args, env, env_params):
        # actor = policy network
        # critic = Q network

        # create actor critic pair
        self.actor  = Actor(env_params, True)
        self.critic = Critic(env_params, True)

        # create target networks which lag the original networks
        self.actor_target = Actor(env_params, True)
        self.critic_target = Critic(env_params, True)

        # loading main params into target for the first time
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # using Adam optimizer
        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic_optimizer  = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        if args.cuda :
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.args = args

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs_dim'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal_dim'], default_clip_range=self.args.clip_range)

        actor_model_path = os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"model.pt"))
        if os.path.isfile(actor_model_path):
            self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, actor_model = torch.load(actor_model_path, map_location=lambda storage, loc: storage)

            self.actor.load_state_dict(actor_model)
            print(color.BOLD + color.YELLOW + "[*] Loaded actor from pretrained model"+ color.END)
        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target_her.pth"))):
            self.actor_target.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target_her.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded actor target from last checkpoint"+ color.END)
        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_her.pth"))):
            self.critic.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_her.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded critic from last checkpoint"+ color.END)
        # if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target_her.pth"))):
            self.critic_target.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target_her.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded critic target from last checkpoint"+ color.END)

        self.env = env
        self.env_params = env_params
        self.test_env = gym.make(args.env_name).env
        self.her_object = HER(self.env.compute_reward)


        self.ou_noise = OUNoise(self.env_params['action_dim'], 123)

        self.buffer = ReplayBuffer(self.env_params['obs_dim'], self.env_params['action_dim'], self.env_params['goal_dim'], self.args.buff_size)
        buffer_path = os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"buffer_her.pth"))
        if os.path.isfile(buffer_path) and (os.path.getsize(buffer_path) > 0):
            #self.buffer = torch.load(buffer_path)
            print(color.BOLD + color.YELLOW + "[*] Loaded last saved buffer")

    def _generate_action_with_noise(self, obs, noise):
        self.actor.eval()
        action = self.actor(torch.Tensor(obs.reshape(1,-1)))
        #print("noise: ",noise*self.ou_noise.sample())
        action = action.detach().cpu().numpy().squeeze() + noise*self.ou_noise.sample()
        self.actor.train()
        return np.clip(action, -self.env_params['max_action'], self.env_params['max_action'])

    def _validation(self):
        print(color.BOLD + color.BLUE + "Validating : " + color.END)
        for i in range(10):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            o = o['observation']
            while not (d or (ep_len == self.args.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self._generate_action_with_noise(o, 0))
                o = o['observation']
                ep_ret += r
                ep_len += 1
            print("Episode length : {}, Episode reward : {}".format(ep_len, ep_ret))

    def _concat_inputs(self, o, g):
        obs_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)

        if o.shape[0] == 25:
            inputs = np.concatenate([obs_norm, g_norm])
        else:
            inputs = np.concatenate([obs_norm, g_norm], axis = 1)
        inputs = torch.tensor(inputs, dtype=torch.float32)#.unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _her_util(self, transitions):
        ep_obs, ep_ag, ep_g, ep_actions = transitions
        ep_obs_next = ep_obs[1:, :]
        ep_ag_next  = ep_ag[1:, :]

        buffer_temp = {'obs': ep_obs,
                       'ag': ep_ag,
                       'g': ep_g,
                       'actions': ep_actions,
                       'obs_next': ep_obs_next,
                       'ag_next': ep_ag_next,
                       }

        return self.her_object._apply_hindsight(buffer_temp)

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.squeeze()

        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['max_action'], high=self.env_params['max_action'], \
                                           size=self.env_params['action_dim'])
        # choose if use the random actions
        action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)
        return action

    def train(self):
        original = sys.stdout
        with open('helloworld.txt', 'w') as filehandle:
            sys.stdout = filehandle
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
            print("[*] Starting ")
            print("---------------------------------------------------------------------")
            print()
        sys.stdout = original
        for epoch in range(self.args.epochs):

            print("\n[*] Epoch {} starts".format(epoch))

            episode_reward = 0
            episode   = 0
            done           = False
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            obs_reset      = self.env.reset()
            o  = obs_reset['observation']
            ag = obs_reset['achieved_goal']
            g  = obs_reset['desired_goal']
            print(color.BOLD + color.RED + "Initial goal : " + color.END, g)

            while (episode < 16*self.env._max_episode_steps):
                #self.env.render()

                #print("A new timestep starts")

                #print(o.shape)
                #print(g.shape)
                inputs = self._concat_inputs(o,g)
                action = self._generate_action_with_noise(inputs, self.args.noise_scale)
                action = self._select_actions(action)
                #print("[*] Action : {} ".format(action))

                obs_nextt, reward, done, _ = self.env.step(action)

                o_next = obs_nextt['observation']
                ag_next= obs_nextt['achieved_goal']
                g_next = obs_nextt['desired_goal']
                #print(color.BOLD + color.RED + "New goal : " + color.END, g_next)

                episode += 1
                #print(color.BOLD + color.BLUE + "obs : " + color.END, o_next)
                ep_obs.append(o.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())

                #store experience in buffer
                self.buffer._store(o, o_next, action, reward, g, done, 0) # or type=std

                o  = o_next
                ag = ag_next
                g  = g_next

            ep_obs.append(o.copy())
            ep_ag.append(ag.copy())

            ep_obs     = np.array(ep_obs)
            ep_ag      = np.array(ep_ag)
            ep_g       = np.array(ep_g)
            ep_actions = np.array(ep_actions)

            #print(color.BOLD + color.YELLOW + "Size of buffer : " + color.END, self.buffer.size)

            # performing her
            self.hind_experiences = self._her_util([ep_obs, ep_ag, ep_g, ep_actions])
            #print("hind_experiences")
            #print(self.hind_experiences)
            num_transitions = len(self.hind_experiences['r'])

            obs, g = self.hind_experiences['obs'], self.hind_experiences['g']
            # pre process the obs and g
            self.hind_experiences['obs'], self.hind_experiences['g'] = self._preproc_og(obs, g)
            # update
            self.o_norm.update(self.hind_experiences['obs'])
            self.g_norm.update(self.hind_experiences['g'])
            # recompute the stats
            self.o_norm.recompute_stats()
            self.g_norm.recompute_stats()


            #print(num_transitions)
            for i in range(num_transitions):
                temp_done = 1 if (i == num_transitions-1) else 0
                self.buffer._store(self.hind_experiences['obs'][i],
                                   self.hind_experiences['obs_next'][i],
                                   self.hind_experiences['actions'][i],
                                   self.hind_experiences['r'][i],
                                   self.hind_experiences['g'][i],
                                   temp_done, # done = 1 for last tuple
                                   1  # type 0 for std, 1 for her
                                   )

            #print(color.BOLD + color.YELLOW + "No. of hind experiences : " + color.END, num_transitions)
            print(color.BOLD + color.YELLOW + "Size of buffer : " + color.END, self.buffer.size)


            episode = 0
            h_prop = 0
            hind_prop = np.array([])
            #print(color.BOLD + color.RED + "Network update begins" + color.END)
            while (episode < self.args.timesteps):
                self.actor.train()
                self.critic.train()
                self.actor_target.train()
                self.critic_target.train()

                batch = self.buffer._sample_batch()
                (obs, obs_next, actions, rewards, goals, dones, replay_types) = (torch.Tensor(batch['obs1']),
                                                                                 torch.Tensor(batch['obs2']),
                                                                                 torch.Tensor(batch['action']),
                                                                                 torch.Tensor(batch['reward']),
                                                                                 torch.Tensor(batch['goal']),
                                                                                 torch.Tensor(batch['done']),
                                                                                 (batch['replay_type']))



                # start to do the update


                if self.args.cuda:
                    obs = obs.cuda()
                    obs_next = obs_next.cuda()
                    actions = actions.cuda()
                    rewards = rewards.cuda()
                    goals = goals.cuda()

                # deactivating autograd engine to save memory
                #with torch.no_grad():
                #print("**********************")


                hind_prop = np.append(hind_prop,np.count_nonzero(replay_types)*100/len(replay_types))

                # print("Rewards: ",rewards.mean())
                # print(obs_next.shape)
                # print(goals.shape)

                ##**************************************************
                ## here either call the _concat_inputs function and do an unsqueeze(0) at the actor output since it gives out [m,4] and we need
                obs, goals = self._preproc_og(obs.cpu(), goals.cpu())
                obs_norm = self.o_norm.normalize(obs.detach().numpy())
                g_norm = self.g_norm.normalize(goals.detach().numpy())
                input      = self._concat_inputs(obs_norm, g_norm)

                obs_next, goals = self._preproc_og(obs_next.cpu(), goals.cpu())
                obs_next_norm = self.o_norm.normalize(obs_next.detach().numpy())
                g_next_norm = self.g_norm.normalize(goals.detach().numpy())
                input_next = self._concat_inputs(obs_next_norm, g_next_norm)
                #input_next = np.concatenate([obs_next.cpu(), goals.cpu()], axis = 1)
                #print(input_next.shape)
                #input_next = torch.tensor(input_next, dtype=torch.float32).unsqueeze(0)

                action_next = self.actor_target(input_next)
                #print("[*] Action : {} Action_next : {}".format(action, action_next))

                q_next      = self.critic_target(input_next,action_next).detach()

                bellman_backup = (rewards + self.args.gamma * (1-dones) * q_next).detach()
                q_predicted    =  self.critic(input_next, actions)

                #print("[*] q_pred : {} q_targ : {}".format(q_predicted, bellman_backup))
                # calculating critic losses and updating it
                critic_loss = F.mse_loss(q_predicted, bellman_backup)


                # print(color.BLUE + "Critic loss: {}".format(critic_loss) + color.END)
                # print(color.BLUE + "Actor loss: {}".format(actor_loss) + color.END)

                # updating critic (Q) network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                action      = self.actor(input)#.unsqueeze(0)
                #print("action shape after unsqueezing: ",action.shape)
                actor_loss  = -self.critic(input, action).mean()

                # updating actor (policy) network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                episode += 1
                episode_reward += rewards.mean()

            #print(color.BOLD + color.RED + "Polyak averaging " + color.END)
            # updating target networks with polyak averaging
            for main_params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_params.data.copy_(self.args.polyak*target_params.data + (1-self.args.polyak)*main_params.data)

            for main_params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_params.data.copy_(self.args.polyak*target_params.data + (1-self.args.polyak)*main_params.data)

                #?????
                #o = o_next


            print(color.RED + "[*] Number of episodes : {} Reward : {} H_prop: {}".format(episode, episode_reward/episode, hind_prop.mean())+ color.END)

            original = sys.stdout
            with open('helloworld.txt', 'a') as filehandle:
                sys.stdout = filehandle
                print("Critic loss : ", critic_loss)
                print("Actor loss : " , actor_loss)
                print("Achieved goal : " , obs_nextt['achieved_goal'])
                print("Desired goal : " , obs_nextt['desired_goal'])
                print("[*] Number of episodes : {} Reward : {}".format(episode, episode_reward))
                print("[*] End of epoch ",epoch)
                print("---------------------------------------------------------------------")
                print()
            sys.stdout = original
            # # Save model
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor.state_dict()], \
                       os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"model.pt")))
            #torch.save(self.actor.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"model.pt")))
            torch.save(self.critic.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_her.pth")))
            torch.save(self.actor_target.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target_her.pth")))
            torch.save(self.critic_target.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target_her.pth")))
            #save buffer
            #torch.save(self.buffer, os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"buffer.pth")))


            # Test the performance of the deterministic version of the agent.
            #self._validation()





