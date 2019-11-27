import argparse
from networks import actor_critic
from train import *
from test import *


parser = argparse.ArgumentParser()

parser.add_argument('--env_name',      type=str,   default='FetchSlide-v1', help='Fetch environment name')
parser.add_argument('--epochs',        type=int,   default=50,              help='Fetch environment name')
parser.add_argument('--phase',         type=str,   default='test',          help='train or test')
parser.add_argument('--model_dir',     type=str,   default='./saved_models',help='path to model directory')
parser.add_argument('--test_episodes', type=int,   default=20,              help='number of episodes testing should run')
parser.add_argument('--clip-obs',      type=float, default=200,             help='the clip ratio')
parser.add_argument('--clip-range',    type=float, default=5,               help='the clip range')
parser.add_argument('--lr_actor',      type=float, default=0.001,           help='learning rate for actor')
parser.add_argument('--lr_critic',     type=float, default=0.001,           help='learning rate for critic')

args = parser.parse_args()

if(args.phase == 'train'):
    train_agent(args)
elif (args.phase == 'test'):
    test_agent(args)
else:
    print("Unknown phase. Enter train or test")


