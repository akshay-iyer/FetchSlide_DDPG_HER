import torch
import torch.nn.functional as F
import gym
import os
from mpi4py import MPI
import numpy as np
import sys

from networks.actor_critic import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.idx = 0
        self.size = 0
        self.max_size = size

        self.obs1_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buffer   = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buffer = np.zeros([size, act_dim], dtype=np.float32)
        self.reward_buffer = np.zeros(size           , dtype=np.float32)
        self.done_buffer   = np.zeros(size           , dtype=np.float32)

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

class DDPG:
    def __init__(self, args, env, env_params):
        # actor = policy network
        # critic = Q network

        # create actor critic pair
        self.actor  = Actor(env_params)
        self.critic = Critic(env_params)

        # create target networks which lag the original networks
        self.actor_target = Actor(env_params)
        self.critic_target = Critic(env_params)

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

        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor.pth"))):
            self.actor.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded actor from last checkpoint")
        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target.pth"))):
            self.actor_target.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded actor target from last checkpoint")
        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic.pth"))):
            self.critic.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded critic from last checkpoint")
        if os.path.isfile(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target.pth"))):
            self.critic_target.load_state_dict(torch.load(os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target.pth"))))
            print(color.BOLD + color.YELLOW + "[*] Loaded critic target from last checkpoint")


        self.env = env
        self.env_params = env_params
        self.test_env = gym.make(args.env_name).env



        self.buffer = ReplayBuffer(self.env_params['obs_dim'], self.env_params['action_dim'], self.args.buff_size)
        buffer_path = os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"buffer.pth"))
        if os.path.isfile(buffer_path) and os.path.getsize(buffer_path) > 0:
            self.buffer = torch.load(buffer_path)
            print(color.BOLD + color.YELLOW + "[*] Loaded last saved buffer")

    def generate_action_with_noise(self, obs, noise):
        self.actor.eval()
        action = self.actor(torch.Tensor(obs.reshape(1,-1)))
        action = action.detach().cpu().numpy().squeeze() + noise*np.random.randn(self.env_params['action_dim'])
        self.actor.train()
        return np.clip(action, -self.env_params['max_action'], self.env_params['max_action'])

    def validation(self):
        print(color.BOLD + color.BLUE + "Validating : " + color.END)
        for i in range(10):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            ###o = o['observation']
            while not d and (ep_len < 200):
                # Take deterministic actions at test time (noise_scale=0)

                o, r, d, _ = self.test_env.step(self.generate_action_with_noise(o, 0))
                #0self.env.render()
                ###o = o['observation']
                ep_ret += r
                ep_len += 1

            print("Episode length : {}, Episode reward : {}".format(ep_len, ep_ret))

    def train(self):
        for epoch in range(self.args.epochs):

            print("[*] Epoch {} starts".format(epoch))
            # self.actor.eval()
            # self.critic.eval()

            # # initial random exploration
            # if(step < self.args.start_steps):
            #     action = self.env.action_space.sample()
            # else:
            #     action = self.generate_action_with_noise(o, self.args.noise_scale)

            episode_reward = 0
            episode   = 0
            done           = False

            o      = self.env.reset()
            ###o = obs_reset['observation']

            # action = self.env.action_space.sample()
            # print("Initial action : ",action)
            # # take one step
            # o_next, r, d, _ = self.env.step(action)

            # print(color.BOLD + color.BLUE + "done : " + color.END, d)
            # print(color.BOLD + color.BLUE + "ach goal : " + color.END, o_next['achieved_goal'])
            # print(color.BOLD + color.BLUE + "des goal : " + color.END, o_next['desired_goal'])

            #o_next = o_next['observation']
            #print(color.BOLD + color.BLUE + "obs : " + color.END, o_next)


            # store experience in buffer
            # self.buffer.store(o, o_next, action, r, d)
            # print("Uniques in buffer : ",len(np.unique(self.buffer.obs1_buffer)))
            #
            # episode_reward += r
            # episode_len +=1

            #d = False if episode_len == self.args.max_ep_len else d

            # update observation
            #o=o_next

            while not done and (episode < 300):
                #self.env.render()
                #print ("episode length : {} d : {}".format(episode_len, d))
                #print("---------------------------------------------------------------------")
                #print("[*] Episode {} starts".format(episode))
                self.actor.train()
                self.critic.train()
                self.actor_target.train()
                self.critic_target.train()

                action = self.generate_action_with_noise(o, self.args.noise_scale)
                #print("[*] Action : {} ".format(action))

                o_next, reward, done, _ = self.env.step(action)
                #print(color.BOLD + color.RED + "trg reward : " + color.END, reward)
                episode_reward += reward
                ###o_next = obs_nextt['observation']
                #print(color.BOLD + color.BLUE + "obs : " + color.END, o_next)


                # store experience in buffer
                self.buffer.store(o, o_next, action, reward, done)
                #print("Size of buffer : ",self.buffer.size)


                #for _ in range(episode_len):
                # batch size 32 or 100?
                batch = self.buffer.sample_batch()
                (obs, obs_next, actions, rewards, dones) = (torch.Tensor(batch['obs1']),
                                                            torch.Tensor(batch['obs2']),
                                                            torch.Tensor(batch['action']),
                                                            torch.Tensor(batch['reward']),
                                                            torch.Tensor(batch['done']))

                if self.args.cuda:
                    obs = obs.cuda()
                    obs_next = obs_next.cuda()
                    actions = actions.cuda()
                    rewards = rewards.cuda()

                # deactivating autograd engine to save memory
                #with torch.no_grad():

                action_next = self.actor_target(obs_next)
                #print("[*] Action : {} Action_next : {}".format(action, action_next))

                q_next      = self.critic_target(obs_next,action_next).detach()

                bellman_backup = (rewards + self.args.gamma * (1-dones) * q_next).detach()
                q_predicted    =  self.critic(obs, actions)

                #print("[*] q_pred : {} q_targ : {}".format(q_predicted, bellman_backup))
                # calculating critic losses and updating it
                critic_loss = F.mse_loss(q_predicted, bellman_backup)


                # print(color.BLUE + "Critic loss: {}".format(critic_loss) + color.END)
                # print(color.BLUE + "Actor loss: {}".format(actor_loss) + color.END)

                # updating critic (Q) network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                action      = self.actor(obs)
                actor_loss  = -self.critic(obs, action).mean()

                # updating actor (policy) network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()



                # updating target networks with polyak averaging
                for main_params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_params.data.copy_(self.args.polyak*target_params.data + (1-self.args.polyak)*main_params.data)

                for main_params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_params.data.copy_(self.args.polyak*target_params.data + (1-self.args.polyak)*main_params.data)

                #obs_reset, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
                o = o_next
                episode += 1


            print("[*] Number of episodes : {} Reward : {}".format(episode, episode_reward))


            # print(color.BOLD + color.BLUE + "ach goal : " + color.END, o_next['achieved_goal'])
            # print(color.BOLD + color.BLUE + "des goal : " + color.END, o_next['desired_goal'])
            original = sys.stdout
            with open('helloworld.txt', 'a') as filehandle:
                sys.stdout = filehandle
                print(color.BOLD + color.BLUE + "Critic loss : " + color.END, critic_loss)
                print(color.BOLD + color.BLUE + "Actor loss : " + color.END, actor_loss)
                # print(color.BOLD + color.BLUE + "Achieved goal : " + color.END, obs_nextt['achieved_goal'])
                # print(color.BOLD + color.BLUE + "Desired goal : " + color.END, obs_nextt['desired_goal'])

                print("[*] Number of episodes : {} Reward : {}".format(episode, episode_reward))
                #print("[*] Number of episodes : ",episode)
                print("[*] End of epoch ",epoch)
                print("---------------------------------------------------------------------")
                print()
            sys.stdout = original
            # # Save model
            torch.save(self.actor.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor.pth")))
            torch.save(self.critic.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic.pth")))
            torch.save(self.actor_target.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"actor_target.pth")))
            torch.save(self.critic_target.state_dict(), os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"critic_target.pth")))
            # save buffer
            torch.save(self.buffer, os.path.join(self.args.model_dir, os.path.join(self.args.env_name,"buffer.pth")))

            # Test the performance of the deterministic version of the agent.
            #self.validation()





