#!/usr/bin/env python3

import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def train(env_id, num_timesteps, seed, alg, lr, momentum):
    env = make_mujoco_env(env_id, seed)

    if alg == 'sgd':
        from baselines.acktr.acktr_cont import learn
    elif alg == 'mid':
        from baselines.acktr.acktr_cont_midpoint import learn
    elif alg == 'geo':
        from baselines.acktr.acktr_cont_geo import learn
    else:
        raise ValueError
    nprocs = 4
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        policy = GaussianMlpPolicy(ob_dim, ac_dim, 'pi')

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False, lr=lr, momentum=momentum)

        env.close()


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir='geo/v{}/{}/{}/{}/{}'.format(args.version, args.alg, args.env, args.lr, args.seed))
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, alg=args.alg, lr=args.lr, momentum=args.mom)


if __name__ == "__main__":
    main()
