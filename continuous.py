import gym, math
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv, render, Sensei
from agent import Agent
import tensorflow as tf

tf.set_random_seed(0)

two_pi = np.sqrt(0.5 / np.pi)

def main():
    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('Pendulum-v0')
    # env = gym.make('CartPole-v1')

    continuous = isinstance(env.action_space, gym.spaces.Box)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

    # num_iterations = 600
    # sample_horizon = 2048
    num_iterations = 1
    sample_horizon = 5
    gamma, lam = 0.99, 0.95

    vero = Agent(continuous, ob_dim, ac_dim, n_layers=2)
    regina = Sensei(vero, continuous, ob_dim, ac_dim,
                    # num_epochs=10, batch_size=64,
                    num_epochs=1, batch_size=2,
                    learning_rate=5e-3, epsilon=0.2)

    init = tf.global_variables_initializer()
    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, vero, env, sample_horizon)

        for i in range(num_iterations):
            seg = generator.__next__()
            add_vtarg_adv(seg, lam, gamma)

            adv = seg["adv"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            regina.train_samples(sess, seg["ob"], seg["ac"], adv, seg["vtarg"], seg["log_probs"])

            rewards = np.array(seg["ep_rets"])
            # print(rewards)

            # if i % 10 == 0 or i == num_iterations-1:
            #     print('Iteration {0:3d}: with average rewards {1:5.3f} and std {2:5.2f}'
            #           .format(i, rewards.mean(), rewards.std()))

        # render(sess, vero, env)
        

if __name__ == '__main__':
    main()
