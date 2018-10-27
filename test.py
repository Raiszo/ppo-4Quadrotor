import gym
import tensorflow as tf
import numpy as np
from agent import Agent
from ppo import render, rollouts_generator, Sensei

num_iterations = 2
sample_horizon = 10
# Learning hyperparameters
epochs=2
batch_size=2
learning_rate=3e-4
# GAE params
gamma=0.99
lam=0.95
# PPO specific hyperparameter, not gonna change this :v
epsilon=0.2

def main():
    env = gym.make('LunarLanderContinuous-v2')
    
    continuous = isinstance(env.action_space, gym.spaces.Box)
    print(continuous)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

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
            add_vtarg_adv(seg, lam, gamma)

            adv = seg["adv"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = adv[:, None]

            regina.train_samples(sess, seg["ob"], seg["ac"], adv, seg["vtarg"], seg["log_probs"])

            rewards = np.array(seg["ep_rets"])
            
            if i % 10 == 0 or i == num_iterations-1:
                mean, std = rewards.mean(), rewards.std()
                print('Iteration {0:3d}: with average rewards {1:5.3f} and std {2:5.2f}'
                      .format(i, mean, std))

        render(sess, veronika, env)

if __name__ == '__main__':
    main()
