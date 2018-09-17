import os
import numpy as np
import tensorflow as tf
from policy import build_mlp as mlp


# def rollouts_generator(pi, env, horizon):
def rollouts_generator(sess, pi, env, horizon):
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
    vpreds = np.zeros(horizon, 'float32')

    news = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float32')

    while True:
        # prevac = ac
        # ac, vpred = pi.act(ob)
        ac, vpred = pi.act(sess, ob)

        if t > 0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews, "new": news,
                    "vpred": vpreds, "next_vpred": vpred*(1-new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens }
        
        i = t % horizon
        
        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        news[i] = new

        ob, rew, new, _ = env.step(ac)

        rews[i] = rew

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

        t += 1
