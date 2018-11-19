import gym
import gym_pendrogone
from ppo import render
from os import path as path
# import gym_pendrogone

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act_deterministic(self, obs, sess=None):
        return self.env.action_space.sample()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render random actions in an environment')
    parser.add_argument('env_name', help='env name')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    vero = RandomAgent(env)

    render(vero, env)
    
    
if __name__ == '__main__':
    main()
