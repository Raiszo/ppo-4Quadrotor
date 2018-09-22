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
    std = 0.1
    learning_rate = 0.05

    # Sampled variables
    ob_no = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])
    ac_na = tf.placeholder(tf.float32, shape=[None, env.action_space.shape[0]])
    adv_n = tf.placeholder(tf.float32, shape=[None])
    val_n = tf.placeholder(tf.float32, shape=[None])

    # Target Value function
    t_val = tf.placeholder(tf.float32, shape=[None])
    
    pi = Policy('veronika', ob_no, ac_dim, n_layers=2)

    
    # Gaussian policy loss operations
    mean_na = pi.action
    logprob_n = (ac_na - mean_na) / std**2
    pg_loss = tf.reduce_mean(logprob_n)

    # Value function loss operations
    b_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=t_val, predictions=val_n))
    

    loss = pg_loss + b_loss
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, pi, env, 202)

        for _ in range(100):
            seg = generator.__next__()
            add_vtarg_adv(seg, lam, gamma)

            feed_dict = {
                ob_no: seg["ob"],
                ac_na: seg["ac"],
                adv_n: seg["adv"],
                val_n: seg["vpred"],
                t_val: seg["vtarg"]
            }

            _loss, _ = sess.run([loss, update_op], feed_dict=feed_dict)

            print(seg["rew"].sum())
            print(_loss)

        

if __name__ == '__main__':
    main()
