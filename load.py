import gym
import gym_pendrogone
import json
from ppo import rollouts_generator, add_vtarg_adv, render, Sensei
from agent import Agent
import tensorflow as tf
from os import path as path
# import gym_pendrogone

def load(exp_dir):
    # exp_dir = 'experiments/PPO-00_Pendulum-v0_25-10-2018_20-12-37'
    ckpt_path = path.join(exp_dir, '0/model349.ckpt')

    params_path = path.join(exp_dir, 'params.json')
    assert path.exists(params_path), "params.json must exist at the root of the experiment folder >:v"

    with open(params_path) as f:
        params= json.load(f)
    
    env = gym.make(params["env_name"])
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
    import argparse
    parser = argparse.ArgumentParser(description='Render trainned agents in their environments')
    parser.add_argument('logdir', help='relative path to experiment directory')
    # parser.add_argument('iteration', help='iteration number')
    args = parser.parse_args()

    load(args.logdir)
    
    
if __name__ == '__main__':
    main()
