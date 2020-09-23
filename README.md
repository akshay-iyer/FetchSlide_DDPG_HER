## Instructions to run : 

1. Create a virtual environment and install required dependencies : 
pip3 install -r requirements.txt

2. To test using pretrained model : 
python main.py --phase=test

3. To train model using CPU :
  python main.py --phase=train

4. To train model using GPU (resolving bugs in gpu code): 
  python main.py --phase=train --cuda=True


# Reinforcement Learning for Robot Control
> Train the Fetch Robot to slide the puck to the goal position

The aim of this project is to use reinforcement learning to make the Fetch Robot slide a puck to the goal position on the table. For the same, we implement Vanilla Deep Deterministic Policy Gradient (DDPG) [link to paper](https://arxiv.org/abs/1509.02971) and DDPG with Hindsight Experience Replay (HER) [Link to paper](https://arxiv.org/abs/1707.01495) to make use of failed experiences.

![](header.png)
![alt text](https://github.com/akshay-iyer/FetchSlide_DDPG_HER/rl-demo.gif)

## Requirements

The codebase is implemented in Python 3.7. To install the necessary requirements, run the following commands:

```
pip3 install -r requirements.txt
```

## Environments
OpenAI Gym

## Datasets

The scripts for downloading and loading the MNIST and CIFAR10 datasets are included in the `datasets_loader` folder. These scripts will be called automatically the first time the `main.py` script is run.

## Options

Learning and inference of differentiable kNN models is handled by the `pytorch/run_dknn.py` script which provides the following command-line arguments:

```
  --k INT                 number of nearest neighbors
  --tau FLOAT             temperature of sorting operator
  --nloglr FLOAT          negative log10 of learning rate
  --method STRING         one of 'deterministic', 'stochastic'
  --dataset STRING        one of 'mnist', 'fashion-mnist', 'cifar10'
  --num_train_queries INT number of queries to evaluate during training.
  --num_train_neighbors INT number of neighbors to consider during training.
  --num_samples INT       number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  -resume                 start a new model, instead of loading an older one
```

Learning and inference of quantile-regression models is handled by the `tf/run_median.py` script, which provides the following command-line arguments:

```
  --M INT                 minibatch size
  --n INT                 number of elements to compare at a time
  --l INT                 number of digits in each multi-mnist dataset element
  --tau FLOAT             temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         one of 'vanilla', 'sinkhorn', 'gumbel_sinkhorn', 'deterministic_neuralsort', 'stochastic_neuralsort'
  --n_s INT               number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  --lr FLOAT              initial learning rate
```

Learning and inference of sorting models is handled by the `tf/run_sort.py` script, which provides the following command-line arguments:

```
  --M INT                 minibatch size
  --n INT                 number of elements to compare at a time
  --l INT                 number of digits in each multi-mnist dataset element
  --tau FLOAT             temperature (either of sinkhorn or neuralsort relaxation)
  --method STRING         one of 'vanilla', 'sinkhorn', 'gumbel_sinkhorn', 'deterministic_neuralsort', 'stochastic_neuralsort'
  --n_s INT               number of samples for stochastic methods
  --num_epochs INT        number of epochs to train
  --lr FLOAT              initial learning rate

```

## Examples

_Training dKNN model to classify CIFAR10 digits_

```
cd pytorch
python run_dknn.py --k=9 --tau=64 --nloglr=3 --method=deterministic --dataset=cifar10
```

_Training quantile regression model to predict the median of sets of nine 5-digit numbers_

```
cd tf
python run_median.py --M=100 --n=9 --l=5 --method=deterministic_neuralsort
```

_Training sorting model to sort sets of five 4-digit numbers_

```
cd tf
python run_sort.py --M=100 --n=5 --l=4 --method=deterministic_neuralsort
```

## Meta

Akshay Iyer – [@akshay_iyerr](https://twitter.com/akshay_iyerr) – akshay.iyerr@gmail.com

[Github](https://github.com/akshay-iyer/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
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

