import gym
import os, math, time, inspect
from ppo import rollouts_generator, add_vtarg_adv, render, Sensei
from agent import Agent
import tensorflow as tf
from os import path as path
# import gym_pendrogone

def load(exp_name, env_name, num_iterations, sample_horizon,
         gamma, lam, learning_rate, epsilon, epochs, batch_size):
    exp_dir = 'experiments/PPO-00_Pendulum-v0_25-10-2018_20-12-37'
    ckpt_path = path.join(exp_dir, '0/model249.ckpt')

    print(ckpt_path)

    
    env = gym.make(env_name)
    continuous = isinstance(env.action_space, gym.spaces.Box)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

    vero = Agent(continuous, ob_dim, ac_dim, n_layers=2)
    # regina = Sensei(vero, continuous, ob_dim, ac_dim,
    #                 epochs, batch_size,
    #                 learning_rate, epsilon)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        render(sess, vero, env)


def main():
    args = {
        "batch_size"    :       64,
        "env_name"      :       "Pendulum-v0",
        "epochs"        :       10,
        "epsilon"       :       0.2,
        "exp_name"      :       "PPO-00",
        "gamma" :       0.99,
        "lam"   :       0.95,
        "learning_rate" :       0.0003,
        "num_iterations"        :       150,
        "sample_horizon"        :       2048
    }

    load(**args)

    
if __name__ == '__main__':
    main()
