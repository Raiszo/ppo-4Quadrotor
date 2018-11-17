import os
from math import ceil
import numpy as np
import tensorflow as tf


def rollouts_generator(sess, agent, env, horizon):
    """
    Generator function
    This function will continue generating
    samples as long as __next__() method is called
    """
    t = 0
    ac = env.action_space.sample()
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    
    new = True

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    log_probs = np.array([ac for _ in range(horizon)])
    vpreds = np.zeros(horizon, 'float32')

    news = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float32')

    while True:
        # prevac = ac
        # ac, vpred = pi.act(ob)
        
        ac, vpred, log_prob = agent.act(sess, ob)
        # print(ac)
        """
        Need next_vpred if the batch ends in the middle of an episode, then we need to append
        that value to vpreds to calculate the target Value using TD => V = r + gamma*V_{t+1}
        Else (finished episode) then append justa 0, does not mean that the value is 0
        but the Value target for the last step(T-1) is just the reward => V = r
        """
        if t > 0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews, "new": news,
                    "vpred": vpreds, "next_vpred": vpred*(1-new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                    "log_probs": log_probs }
            ep_rets = []
            ep_lens = []
        
        i = t % horizon

        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        log_probs[i] = log_prob
        news[i] = new

        ob, rew, new, _ = env.step(ac)
        # print(ob, rew)

        rews[i] = rew
        cur_ep_ret += rew
        cur_ep_len += 1

        # if new or (ep_len and i > ep_len):
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

        t += 1

def render(agent, env, sess):
    ob = env.reset()
    done = False

    total_rew = 0
    while not done:
        env.render()
        ac = agent.act_deterministic(ob, sess)

        ob, rew, done, _ = env.step(ac)
        print(rew)
        # total_rew += rew

    # print('Total reward at testig', total_rew)
        
def add_vtarg_adv(seg, lam, gamma):
    T = len(seg["ob"])
    new = np.append(seg["new"], 0)
    seg["adv"] = gae_adv = np.empty(T, 'float32')
    seg["vtarg"] = td_v = np.empty(T, 'float32')
    
    vpred = np.append(seg["vpred"], seg["next_vpred"])

    last_gae = 0
    for t in reversed(range(T)):
        # check this, when is_terminal = 1-new[t], everything crushes like crazy
        is_terminal = 1-new[t+1]
        delta = - vpred[t] + (is_terminal * gamma * vpred[t+1] + seg["rew"][t])
        gae_adv[t] = last_gae = delta + gamma*lam*last_gae*is_terminal

        td_v[t] = is_terminal * gamma * vpred[t+1] + seg["rew"][t]


class Sensei():
    def __init__(self, agent, continuous, ob_dim, ac_dim,
                 num_epochs, batch_size,
                 learning_rate, epsilon=0.2):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.agent = agent
        
        # Sampled variables
        with tf.variable_scope('placeholders'):
            ac_args = {
                "shape": [None, ac_dim] if continuous else [None],
                "dtype": tf.float32 if continuous else tf.int32
            }
            self.ac_na = ac_na = tf.placeholder(name='actions', **ac_args)
            self.log_p = log_p = tf.placeholder(name='old_log_probs', **ac_args)
            self.adv_n = adv_n = tf.placeholder(shape=[None, 1], name='advantages', dtype=tf.float32)
            self.t_val = t_val = tf.placeholder(shape=[None], name='target_value', dtype=tf.float32)


        with tf.variable_scope('loss/surrogate'):
            ratio = tf.exp(agent.dist.log_prob(ac_na) - log_p)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)

            surrogate_min = tf.minimum(ratio*adv_n, clipped_ratio*adv_n)
            self.surrogate = tf.reduce_mean(surrogate_min)

        with tf.variable_scope('loss/value_f'):
            v_loss = tf.losses.mean_squared_error(labels=t_val, predictions=agent.vpred)
            self.v_loss = tf.reduce_mean(v_loss)

        with tf.variable_scope('loss/entropy'):
            self.ent_loss = tf.reduce_mean(agent.dist.entropy())

        self.loss = - self.surrogate + 0.5*self.v_loss - 0.01*self.ent_loss

        gradient_clip = 40
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(self.loss, tf.trainable_variables())
        grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.train_op = optimizer.minimize(self.loss)

        # self.variables = [self.train_op, ratio, clipped_ratio, log_p, agent.dist.log_prob(ac_na)]
        self.variables = [self.train_op]

    def train_samples(self, sess, obs, acs, advs, val, log_probs):
        batch_size = self.batch_size
        size = obs.shape[0]
        train_indicies = np.arange(size)
        
        for _ in range(self.num_epochs):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]

                feed_dict = {
                    self.agent.state: obs[idx, :],
                    self.ac_na: acs[idx, :],
                    self.log_p: log_probs[idx, :],
                    self.adv_n: advs[idx],
                    self.t_val: val[idx],
                }

                stuff = sess.run(self.variables, feed_dict)
                # print(stuff)
                # print(stuff[1])
                # print(stuff[2])
                # print(stuff[3])
                # print(stuff[4])
        #         print('end batch')
        #     print('end epoch')
        # print('end ite')

        
