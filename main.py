import gym
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv, render
from agent import Agent
import tensorflow as tf

tf.set_random_seed(0)

def main():
    # env = gym.make('Pendulum-v0')
    env = gym.make('CartPole-v1')

    continuous = isinstance(env.action_space, gym.spaces.Box)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if continuous else env.action_space.n
        
    
    gamma, lam = 0.99, 0.95
    std = 0.1
    learning_rate = 5e-3
    epsilon = 0.2

    # Sampled variables
    ob_no = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)
    ac_na = tf.placeholder(shape=[None, ac_dim], name="actions", dtype=tf.float32) \
        if continuous else \
           tf.placeholder(shape=[None], name="actions", dtype=tf.int32)
    adv_n = tf.placeholder(shape=[None], dtype=tf.float32)
    val_n = tf.placeholder(shape=[None], dtype=tf.float32)

    # Target Value function
    t_val = tf.placeholder(shape=[None], dtype=tf.float32)

    # print(ac_dim)
    rla = Agent('veronika', ob_no, ac_dim, continuous, n_layers=2)

    
    # Gaussian policy loss operations
    # mean_na = pi.action
    # logprob_n = (ac_na - mean_na) / std**2
    # pg_loss = tf.reduce_mean(logprob_n)
    
    # with tf.variable_scope('losses'):
    #     log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ac_na, logits=pi.logits)
    #     pg_loss = tf.reduce_mean(adv_n * log_prob, name='pg_loss')
    #     # Value function loss operations
    #     v_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=t_val, predictions=val_n), name='v_loss')
    #     loss = pg_loss + v_loss
    
    # This only may work for the discrete case
    # probabilities of actions which agent took with policy
    act_probs = rla.pi.logits * tf.one_hot(indices=ac_na, depth=ac_dim)
    act_probs = tf.reduce_sum(act_probs, axis=1)

    # probabilities of actions which agent took with old policy
    act_probs_old = rla.old_pi.logits * tf.one_hot(indices=ac_na, depth=ac_dim)
    act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
    
    with tf.variable_scope('loss/surrogate'):
        ratio = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)
        
        surrogate = tf.minimum(ratio*adv_n, clipped_ratio*adv_n)
        surrogate = - tf.reduce_mean(surrogate)
        
    with tf.variable_scope('loss/value_f'):
        v_loss = tf.losses.mean_squared_error(labels=t_val, predictions=val_n)
        v_loss = tf.reduce_mean(v_loss)

    with tf.variable_scope('loss'):
        loss = surrogate + v_loss
        
    
    gradient_clip = 40
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = tf.gradients(loss, rla.pi_vars)
    # print(pi.policy_vars)
    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
    grads_and_vars = list(zip(grads, rla.pi_vars))
    train_op = optimizer.apply_gradients(grads_and_vars)
    
    
    init = tf.global_variables_initializer()
    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, rla, env, 5)

        for _ in range(2):
            seg = generator.__next__()
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

            for _ in range(5):
                _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)

            rla.save_policy(sess)
            print(_loss)
            print(seg["ep_rets"])
            # print(sum(seg["ep_rets"]) / len(seg["ep_rets"]))

        # render(sess, pi, env)
        

if __name__ == '__main__':
    main()
