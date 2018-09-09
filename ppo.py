import os
import numpy as np
import tensorflow as tf
from policy import build_mlp as mlp


def rollouts_generator(pi, env, horizon, steps):
    ac = env.action_space.sample()
    ob = env.reset()

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    vpreds = np.zeros(horizon, 'float32')

    rews = np.zeros(horizon, 'float32')

    for t in range(steps):
        # prevac = ac
        ac, vpred = pi.act(ob)

        if t>0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews,
                    "vpred": vpreds, "next_vpred": vpred,
                    "ep_len": 0 }
        
        i = t % horizon
        
        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred

        ob, rew, done = env.step(ac)

        rews[i] = rew
