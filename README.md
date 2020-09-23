# Reinforcement Learning for Robot Control
> Train the Fetch Robot to slide/push/pick-and-place the puck to the goal position

The aim of this project is to use reinforcement learning to make the Fetch Robot perform different tasks like pushing/sliding/picking and placing a puck at the goal position on the table. For the same, I implemented Vanilla Deep Deterministic Policy Gradient (DDPG) [link to paper](https://arxiv.org/abs/1509.02971) and DDPG with Hindsight Experience Replay (HER) [Link to paper](https://arxiv.org/abs/1707.01495) to make use of failed experiences.

I created two action-critic pairs, one lagging behind the other. Both the actor and the critic are created as fully connected networks with 4 layers and 256 hidden units each. The networks are defined in `networks/actor-critic.py`. The network contains ReLU activation function for all layers while the last layer has tanH activation. 

The actor is fed the `observation` and `goal` concatenated while the critic is fed the concatenation of `observation`, `goal`, and the `action`. The actor predicts the `actions` while the critic predicts the `q-value`. The networks are implemented using the PyTorch framework and were run for 7000 epochs each with 800 timesteps on a Nvidia 1080Ti GPU. 

![](header.png)
![alt text](rl_demo.gif)

## Requirements

The codebase is implemented in Python 3.7. To install the necessary requirements, run the following commands:

If you use the python shipped virtual environments:
```
python3 -m venv <your_env_name>
source your_env_name/bin/activate
pip3 install -r requirements.txt
```

If you use conda:
```
conda create <your_env_name>
conda activate your_env_name 
conda install --yes --file requirements.txt
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

## Environments
To perform the RL experiments, the wonderful OpenAI Gym [link](https://gym.openai.com/envs/#robotics)

## Options

Training and inference of RL models to perform tasks on the various Fetch environments is handled by the `main.py` script which provides the following command-line arguments

```
  --env_name             Fetch environment name
  --epochs               number of epochs
  --timesteps            number of iterations of network update
  --start_steps          initial number of steps for random exploration
  --max_ep_len           maximum length of episode
  --buff_size            size of replay buffer
  --phase                train or test
  --model_dir            path to model directory
  --test_episodes        number of episodes testing should run
  --clip-obs             the clip ratio
  --clip-range           the clip range
  --lr_actor             learning rate for actor
  --lr_critic            learning rate for critic
  --noise_scale          scaling factor for gaussian noise on action
  --gamma                discount factor in bellman equation
  --polyak               polyak value for averaging
  --cuda                 whether to use GPU
  --her                  whether to use HER
```

## Examples

_Training model on CPU on the Fetch Environment <EnvName>_

```
python main.py --phase=train --env_name=EnvName --cuda=False 
```

_Testing using pretrained model_

```
python main.py --phase=test
```

## Experiments

I performed the following variations:
- Implemented Vanilla DDPG tested on Pendulum-v0 and FetchPush
- Implemented DDPG with HER using 'final' strategy for HER sampling 
- Implemented Gaussian noise and OU noise
- Tried with Input Normalization 
- Initialization using pre-trained weights


## Meta

Akshay Iyer – [@akshay_iyerr](https://twitter.com/akshay_iyerr) – akshay.iyerr@gmail.com

[Portfolio](https://akshay-iyer.github.io/)

## Contributing

1. Fork it (<https://github.com/akshay-iyer/FetchSlide_DDPG_HER/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki

