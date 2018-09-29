import gym
import numpy as np
from ppo import rollouts_generator, add_vtarg_adv, render
from policy import Policy
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
    pi = Policy('veronika', ob_no, ac_dim, continuous, n_layers=2)

    
    # Gaussian policy loss operations
    # mean_na = pi.action
    # logprob_n = (ac_na - mean_na) / std**2
    # pg_loss = tf.reduce_mean(logprob_n)
    
    with tf.variable_scope('losses'):
        pg_loss = tf.reduce_mean(adv_n * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ac_na, logits=pi.logits), name='pg_loss')
        # Value function loss operations
        v_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=t_val, predictions=val_n), name='v_loss')
        loss = pg_loss + v_loss
    
    
    gradient_clip = 40
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    train_op = optimizer.apply_gradients(grads_and_vars)
    
    
    init = tf.global_variables_initializer()
    # gen = generator.__next__()
    with tf.Session() as sess:
        sess.run(init)
        
        generator = rollouts_generator(sess, pi, env, 202)

        for _ in range(100):
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

            _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            print(sum(seg["ep_rets"]) / len(seg["ep_rets"]))

        render(sess, pi, env)
        

if __name__ == '__main__':
    main()
