import os
import numpy as np
import tensorflow as tf
from policy import build_mlp as mlp


def rollouts_generator(pi, env, horizon):
    """
    Generator function
    This function will continue generating
    samples as long as __next__() method is called
    """
    
    ac = env.action_space.sample()
    ob = env.reset()

    new = True

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    vpreds = np.zeros(horizon, 'float32')

    news = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')

    while True:
        # prevac = ac
        ac, vpred = pi.act(ob)

        if t>0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews,
                    "vpred": vpreds, "next_vpred": vpred,
                    "ep_len": 0 , "new": news }
        
        i = t % horizon
        
        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        news[i] = new

        ob, rew, new, _ = env.step(ac)

        rews[i] = rew

        t += 1
