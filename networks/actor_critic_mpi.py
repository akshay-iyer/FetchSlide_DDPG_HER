import torch
import torch.nn as nn
import torch.nn.functional as F

# actor network: takes in [obs, goal] and returns action. hence it is learning a policy
# policy network.
# Input(obs 25 + goal 3) > FC1-ReLU > FC2-ReLU > FC3-ReLU > FC4-tanh > (actions 4*1 vector) * max_action (=1)
class Actor(nn.Module):
    def __init__(self, env_params):
        super().__init__()

        self.fc1 = nn.Linear(env_params['obs_dim'] + env_params['goal_dim'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action_dim'])
        self.max_action = env_params['max_action']

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = torch.tanh(self.action_out(out))

        actions = self.max_action*out

        return actions

# critic network: : computes Q(s,a) - takes in s,a and gives out Q-value
# Q network
# Input(obs 25 + goal 3 + action 4) > FC1-ReLU > FC2-ReLU > FC3-ReLU > q (scalar)
class Critic(nn.Module):
    def __init__(self, env_params):
        super().__init__()

        self.fc1 = nn.Linear(env_params['obs_dim'] + env_params['goal_dim'] + env_params['action_dim'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        out = nn.ReLU(self.fc1(x))
        out = nn.ReLU(self.fc2(out))
        out = nn.ReLU(self.fc3(out))
        q   = self.fc4(out)

        return q

