import gym
import os, math, time, inspect
from multiprocessing import Process
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv, render, Sensei
from utils import get_session
from agent import Agent
import tensorflow as tf
# import gym_pendrogone
from logger import Logger as Logz
import plotter

def experiment(n_experiments, **args):
    initial_seed = 0
    # n_experiments = 4
    # print(args)
    
    if not(os.path.exists('experiments')):
        os.makedirs('experiments')
    
    experiment_dir = args["exp_name"] + '_' + args["env_name"] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    experiment_dir = os.path.join('experiments', experiment_dir)
    
    if not(os.path.exists(experiment_dir)):
        os.makedirs(experiment_dir)
        
    # args = inspect.getargspec(experiment)[0]
    # locals_ = locals()
    # params = {k: locals_[k] if k in locals_ else None for k in args}
    Logz.save_params(experiment_dir, args)

    processes = []
    for e in range(n_experiments):
        seed = initial_seed + 10*e
        log_dir = os.path.join(experiment_dir,'%d'%seed)
        def train_func():
            train_process(log_dir=log_dir, seed=seed, **args)

        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)

    for i in processes:
        p.join()
    print('finished trainning :3')
    return experiment_dir
    
    

def train_process(log_dir, exp_name, env_name, num_iterations, sample_horizon,
                  gamma, lam, learning_rate, epsilon, epochs, batch_size, seed):
    tf.set_random_seed(seed)
    
    env = gym.make(env_name)
    continuous = isinstance(env.action_space, gym.spaces.Box)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n

    logger = Logz(log_dir)
    
    vero = Agent(continuous, ob_dim, ac_dim, n_layers=2)
    regina = Sensei(vero, continuous, ob_dim, ac_dim,
                    epochs, batch_size,
                    learning_rate, epsilon)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    save_path = log_dir
    with get_session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, vero, env, sample_horizon)

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

                logger.log_tabular("Iteration", i)
                logger.log_tabular("AverageReturn", mean)
                logger.log_tabular("StdReturn", std)
                logger.log_tabular("MaxReturn", np.max(rewards))
                logger.log_tabular("MinReturn", np.min(rewards))
                logger.dump_tabular()
                # print('Iteration {0:3d}: with average rewards {1:5.3f} and std {2:5.2f}'
                #       .format(i, mean, std))

            if i % 50 == 0 or i == num_iterations-1:
                saver.save(sess=sess, save_path=save_path+"/model%d"%i+".ckpt")
                print('saving on iteration: %d'%i)

        # render(sess, vero, env)

def main():
    # Leaving everything in a single function for easy later CLI argument parsing
    experiment_params = dict(
        exp_name='PPO-01',
        # env_name='Pendulum-v0',
        env_name='LunarLanderContinuous-v2',
        num_iterations=250,
        sample_horizon=2048,
        # Learning hyperparameters
        epochs=10, batch_size=64, learning_rate=3e-4,
        # GAE params
        gamma=0.99, lam=0.95,
        # PPO specific hyperparameter, not gonna change this :v
        epsilon=0.2,
    )
    exp_dir = experiment(n_experiments=1, **experiment_params)
    data = plotter.get_datasets(exp_dir)
    plotter.plot_data(data, os.path.join(exp_dir, 'plot4this.png'))
    
    
if __name__ == '__main__':
    main()
