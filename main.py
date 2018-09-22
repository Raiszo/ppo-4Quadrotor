import gym
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv
from policy import Policy
import tensorflow as tf

tf.set_random_seed(0)

def main():
    env = gym.make('Pendulum-v0')
    ac_dim = env.action_space.shape[0]
    
    gamma, lam = 0.99, 0.95
    std = 0.2

    ob_no = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])
    ac_na = tf.placeholder(tf.float32, shape=[None, env.action_space.shape[0]])
    adv_n = tf.placeholder(tf.float32, shape=[None])
    
    pi = Policy('veronika', ob_no, ac_dim, n_layers=2)

    
    # Sampling operation
    sy_mean = pi.action
    sy_logprob_n = (ob_no - sy_mean) / std**2

    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        generator = rollouts_generator(sess, pi, env, 202)
        seg = generator.__next__()
        add_vtarg_adv(seg, lam, gamma)

        

if __name__ == '__main__':
    main()
