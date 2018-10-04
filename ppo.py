import os
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
    # cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    # ep_lens = [] # lengths of ...
    
    new = True

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    vpreds = np.zeros(horizon, 'float32')

    news = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float32')

    while True:
        # prevac = ac
        # ac, vpred = pi.act(ob)
        
        ac, vpred = agent.act(sess, ob)
        """
        Need next_vpred if the batch ends in the middle of an episode, then we need to append
        that value to vpreds to calculate the target Value using TD => V = r + gamma*V_{t+1}
        Else (finished episode) then append justa 0, does not mean that the value is 0
        but the Value target for the last step(T-1) is just the reward => V = r
        """
        if t > 0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews, "new": news,
                    "vpred": vpreds, "next_vpred": vpred*(1-new),
                    "ep_rets" : ep_rets }
        
        i = t % horizon

        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        news[i] = new

        ob, rew, new, _ = env.step(ac)

        rews[i] = rew
        cur_ep_ret += rew

        if new:
            ep_rets.append(cur_ep_ret)
            # ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            # cur_ep_len = 0
            ob = env.reset()

        t += 1

def render(sess, agent, env):
    ob = env.reset()
    done = False

    total_rew = 0
    while not done:
        env.render()
        ac, v = agent.act(sess, ob)

        ob, rew, done, _ = env.step(ac)
        total_rew += rew

    print('Total reward at testig', total_rew)
        
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

def multi_normal(sy_means, std):
    # sy_means should be of shape [None, env.action_space.shape[0]]
    # so, the for loop will be evaluated correctly
    num_normals = sy_means.get_shape().as_list()[1]
    tensors = []
    for i in range(num_normals):
        tensors.append(tf.random_normal(tf.shape(sy_means)))
    
    samples = tf.concat(tensors, 1)

    return samples * std + sy_means
