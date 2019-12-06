import argparse
from networks import actor_critic
from train import *
from test_hopper import *


parser = argparse.ArgumentParser()

parser.add_argument('--env_name',       type=str,   default='FetchSlide-v1', help='Fetch environment name')
parser.add_argument('--epochs',         type=int,   default=1000,             help='Number of epochs')
parser.add_argument('--steps_in_epoch', type=int,   default=10000,            help='the times to collect samples per epoch')
parser.add_argument('--start_steps',    type=int,   default=10000,           help='initial number of steps for random exploration')
parser.add_argument('--max_ep_len',     type=int,   default=1000,            help='maximum length of episode')
parser.add_argument('--buff_size',      type=int,   default=int(1e6),        help='size of replay buffer')
parser.add_argument('--phase',          type=str,   default='test',          help='train or test')
parser.add_argument('--model_dir',      type=str,   default='./saved_models',help='path to model directory')
parser.add_argument('--test_episodes',  type=int,   default=50,              help='number of episodes testing should run')
parser.add_argument('--clip-obs',       type=float, default=200,             help='the clip ratio')
parser.add_argument('--clip-range',     type=float, default=5,               help='the clip range')
parser.add_argument('--lr_actor',       type=float, default=0.0001,          help='learning rate for actor')
parser.add_argument('--lr_critic',      type=float, default=0.001,           help='learning rate for critic')
parser.add_argument('--noise_scale',    type=float, default=0.1,             help='scaling factor for gaussian noise on action')
parser.add_argument('--gamma',          type=float, default=0.99,            help='discount factor in bellman equation')
parser.add_argument('--polyak',         type=float, default=0.995,           help='polyak value for averaging')
parser.add_argument('--cuda',           type=bool,  default=False,            help='whether to use GPU')

args = parser.parse_args()

if(args.phase == 'train'):
    train_agent(args)
elif (args.phase == 'test'):
    test_agent(args)
else:
    print("Unknown phase. Enter train or test")


