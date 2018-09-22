import gym
import tensorflow as tf
import numpy as np
from policy import Policy

def main():
    env = gym.make('Pendulum-v0')
    ac_dim = env.action_space.shape[0]
    horizon = 5

    # Sampled variables
    ob_no = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])
    
    pi = Policy('veronika', ob_no, ac_dim, n_layers=2)

    
    ob = env.reset()
    obs = np.array([ob for _ in range(horizon)])
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(horizon):
            ac, vpred = pi.act(sess, ob)
            print(ac)

            obs[i] = ob

        print(obs.shape)
        s_ac, ac = sess.run([pi.sample_action, pi.action], feed_dict={pi.state: obs})
        print(s_ac, ac)

if __name__ == '__main__':
    main()
