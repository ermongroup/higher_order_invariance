
# Accelerating Natural Gradient with Higher-Order Invariance

This repo contains necessary code for reproducing results in the paper [Accelerating Natural Gradient with Higher-Order Invariance](https://arxiv.org/pdf/1803.01273.pdf), ICML 2018, Stockholm, Sweden.

by [Yang Song](http://yang-song.github.io/), [Jiaming Song](http://tsong.me/), and [Stefano Ermon](https://cs.stanford.edu/~ermon/), Stanford AI Lab.

---

In this work, we propose to use midpoint integrators and geodesic corrections to improve the invariance of natural gradient optimization. With our methods, we are able to get accelerated convergence for deep neural network training and higher sample efficiency for deep reinforcement learning.

## Dependencies

The synthetic experiments and deep reinforcement learning experiments are implemented in `Python 3` and `TensorFlow`. Deep neural network training experiments are coded in `MATLAB 2015b`. In order to run deep reinforcement learning experiments, the users need to obtain a valid license of [MuJoCo](http://www.mujoco.org/). 


## Running Experiments

### Invariance
We can observe that for a simple objective where ODEs can be solved accurately, midpoint integrators and geodesic correction methods give much more invariant optimization trajectories than vanilla natural gradient.

To reproduce Figure 2 in the paper, run

```
python synth/gamma_experiment.py
```

### Training Deep Neural Networks
Training deep autoencoders and classifiers on CURVES, MNIST, and FACES datasets. Code was based on [James Martens](http://www.cs.toronto.edu/~jmartens/index.html)' MATLAB implementation for _Deep Learning via Hessian-free Optimization_. ([original code](http://www.cs.toronto.edu/~jmartens/docs/HFDemo.zip))

To download all datasets, run

```
cd mat/
wget www.cs.toronto.edu/~jmartens/mnist_all.mat
wget www.cs.toronto.edu/~jmartens/newfaces_rot_single.mat
wget www.cs.toronto.edu/~jmartens/digs3pts_1.mat
```
Then launch `MATLAB` in directory `mat/`. Experiments can be run by calling

```
nnet_experiments(dataset, algorithm, runName)
```
where the options are listed as follows

* `dataset`: a string. Can be 'CURVES', 'MNIST', 'FACES', or 'MNIST_classification'
* `algorithm`: a string. Can be 'ng', 'geo', 'mid', 'geo_faster' or 'adam'
* `runName`: a string. The name of log files.

### Model-Free Reinforcement Learning over Continuous Control

Model-free reinforcement learning over continuous control tasks for [ACKTR](https://arxiv.org/abs/1708.05144). Based on [OpenAI Baselines](https://github.com/openai/baselines). Our code can be installed by running

```
pip install -e rl/
```

The following are usages for running various RL algorithms tested in the paper

* ACKTR

```
python -m baselines.acktr.run_mujoco --env=Walker2d-v2 --seed=1 --mom=0.0 --lr=0.03 --alg=sgd
```

* Midpoint Integrator

```
python -m baselines.acktr.run_mujoco --env=Walker2d-v2 --seed=1 --mom=0.0 --lr=0.03 --alg=mid
```

* Geodesic Correction

```
python -m baselines.acktr.run_mujoco --env=Walker2d-v2 --seed=1 --mom=0.0 --lr=0.03 --alg=geo
```

## Citation
If you find the idea or code useful for your research, please consider citing our [paper](https://arxiv.org/pdf/1803.01273.pdf):

```
@inproceedings{song2018accelerating,
  title={Accelerating Natural Gradient with Higher-Order Invariance},
  author={Song, Yang and Song, Jiaming and Ermon, Stefano},
  booktitle = {International Conference on Machine Learning (ICML)},
  year={2018},
}
```
