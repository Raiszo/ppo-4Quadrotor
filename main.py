import gym
import numpy as np
from ppo import rollouts_generator

class random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

def main():
    env = gym.make('Pendulum-v0')
    pi = random_policy(env)

    
    generator = rollouts_generator(pi, env, 2048)

    gen = generator.__next__()
    

if __name__ == '__main__':
    main()
