import gym
import gym_pendrogone
import tensorflow as tf
import numpy as np
from agent import Agent
from ppo import rollouts_generator, add_vtarg_adv, render, Sensei

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_iterations = 250
sample_horizon = 2048
# Learning hyperparameters
epochs=10
batch_size=64
learning_rate=3e-4
# GAE params
gamma=0.99
lam=0.95
# PPO specific hyperparameter, not gonna change this :v
epsilon=0.2


def main():
    # env = gym.make('PendrogoneZero-v0')
    env = gym.make('DroneZero-v0')
    
    continuous = isinstance(env.action_space, gym.spaces.Box)
    # print(continuous)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

    # print('ob_dim', ob_dim)
    # print('ac_dim', ac_dim)
    

    veronika = Agent(continuous, ob_dim, ac_dim, n_layers=2)
    regina = Sensei(veronika, continuous, ob_dim, ac_dim,
                    epochs, batch_size,
                    learning_rate, epsilon)

    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        generator = rollouts_generator(sess, veronika, env, sample_horizon)

        for i in range(num_iterations):
            seg = generator.__next__()
            # print(seg["rew"])
            add_vtarg_adv(seg, lam, gamma)

            adv = seg["adv"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv[:, None]

            regina.train_samples(sess, seg["ob"], seg["ac"], adv, seg["vtarg"], seg["log_probs"])

            rewards = np.array(seg["ep_rets"])
            
            if i % 10 == 0 or i == num_iterations-1:
                if rewards.shape[0] > 0:
                    mean, std = rewards.mean(), rewards.std()
                    print('Iteration {0:3d}: reward:  m{1:5.3f}, std{2:5.2f}, ep_len: {3:5.2f}'
                          .format(i, mean, std, np.mean(seg["ep_lens"])))

        render(sess, veronika, env)

if __name__ == '__main__':
    main()
