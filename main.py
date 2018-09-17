import gym
import numpy as np
from ppo import rollouts_generator
from policy import Policy
import tensorflow as tf

def main():
    env = gym.make('Pendulum-v0')

    obs_ph = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])
    ac_dim = env.action_space.shape[0]
    
    pi = Policy('veronika', obs_ph, ac_dim, 2)
    # generator = rollouts_generator(pi, env, 2048)

    # gen = generator.__next__()

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset()

        for i in range(10):
            ac, v = pi.act(sess, obs)
            obs, rew, done, _ = env.step(ac)

if __name__ == '__main__':
    main()
