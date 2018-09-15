import gym
import numpy as np
from ppo import rollouts_generator

def main():
    env = gym.make('Pendulum-v0')
    generator = rollouts_generator(env, 2048)

    gen = generator.__next__()
    

if __name__ == '__main__':
    main()
