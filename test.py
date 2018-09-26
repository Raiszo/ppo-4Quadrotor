import gym
import tensorflow as tf
import numpy as np
from policy import Policy

def main():
    env = gym.make('CartPole-v1')
    continuous = isinstance(env.action_space, gym.spaces.Box)

    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

    horizon = 5

    # Sampled variables
    ob_no = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]])

    pi = Policy('veronika', ob_no, ac_dim, continuous, n_layers=2)

    
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
        s_ac, ac = sess.run([pi.sample_action, pi.logits], feed_dict={pi.state: obs})
        print(s_ac, ac)

if __name__ == '__main__':
    main()
