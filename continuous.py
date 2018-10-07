import gym
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv, render
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
        
    
    gamma, lam = 0.9, 0.95
    learning_rate = 5e-3
    epsilon = 0.2
    # epochs = 1
    # num_ite = 1
    # sample_size = 2048
    epochs = 10
    num_ite = 50
    sample_size = 2048

    # Sampled variables
    with tf.variable_scope('placeholders'):
        ob_no = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)
        ac_na = tf.placeholder(shape=[None, ac_dim], name="actions", dtype=tf.float32) \
            if continuous else \
               tf.placeholder(shape=[None], name="actions", dtype=tf.int32)
        adv_n = tf.placeholder(shape=[None], dtype=tf.float32)
        val_n = tf.placeholder(shape=[None], dtype=tf.float32)
        # Target Value function
        t_val = tf.placeholder(shape=[None], dtype=tf.float32)

    # TODO: assing yet another with variable_scope inside agent
    vero = Agent(ob_no, ac_dim, continuous, n_layers=2)

    
    # Continuous case:
    # log_probs = vero.dist.log_prob(ac_na)
    # log_probs = tf.squeeze(log_probs, axis=1)
    probs = vero.dist.prob(ac_na)
    # probs = tf.squeeze(probs, axis=1)
    # old_log_probs = vero.old_dist.log_prob(ac_na)
    # old_log_probs = tf.squeeze(old_log_probs, axis=1)
    old_probs = vero.old_dist.prob(ac_na)
    # old_probs = tf.squeeze(old_probs, axis=1)
    
    with tf.variable_scope('loss/surrogate'):
        # ratio = tf.exp(log_probs - old_log_probs)
        ratio = probs / (old_probs + 1e-8)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)

        surrogate_min = tf.minimum(ratio, clipped_ratio)
        surrogate_mean = tf.reduce_mean(surrogate_min)
        surrogate = tf.reduce_mean(surrogate_min*adv_n)
        
    with tf.variable_scope('loss/value_f'):
        v_loss = tf.losses.mean_squared_error(labels=t_val, predictions=vero.vpred)
        v_loss = tf.reduce_mean(v_loss)

    with tf.variable_scope('loss/entropy'):
        ent_loss = tf.reduce_mean(vero.dist.entropy())

        
    loss = - surrogate + 0.5*v_loss - 0.01*ent_loss
    # loss = - surrogate + 0.5*v_loss
        
    
    gradient_clip = 40
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = tf.gradients(loss, vero.vars)
    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
    grads_and_vars = list(zip(grads, vero.vars))
    train_op = optimizer.apply_gradients(grads_and_vars)

    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.minimize(loss, var_list=vero.vars)

    init = tf.global_variables_initializer()
    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, vero, env, sample_size)

        for i in range(num_ite):
            seg = generator.__next__()
            # From the beginning, the old policy is equal to the current policy
            vero.save_policy(sess)
            add_vtarg_adv(seg, lam, gamma)

            adv = seg["adv"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            
            feed_dict = {
                ob_no: seg["ob"],
                ac_na: seg["ac"],
                adv_n: adv,
                val_n: seg["vpred"],
                t_val: seg["vtarg"]
            }
            # if np.isnan(ac_na).any():
            #     print('fuck my life')

            total_loss = 0
            for _ in range(epochs):
                # _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                _stuff = sess.run([surrogate_mean, v_loss, loss, train_op], feed_dict=feed_dict)
                total_loss += _stuff[0]

                # print(_stuff[0], _stuff[1], _stuff[2])
            
            rewards = np.array(seg["ep_rets"])
            actions = np.array(seg["ac"])

            if i % 5 == 0 or i == num_ite:
                print(total_loss / epochs)
                print(rewards.mean(), rewards.std())
                print(actions.mean(), actions.std())


        render(sess, vero, env)
        

if __name__ == '__main__':
    main()
