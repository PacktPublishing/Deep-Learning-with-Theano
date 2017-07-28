from __future__ import print_function
from keras.optimizers import RMSprop
import gym
from scipy.misc import imresize
from skimage.color import rgb2gray
import numpy as np
import argparse
from model import build_networks

parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='Breakout-v0', help='Name of openai gym environment')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation')
parser.add_argument('--model', help='File with weights for model')
args = parser.parse_args()

env = gym.make(args.game)
if args.evaldir:
    env.monitor.start(args.evaldir)

screen=(84, 84)
input_depth = 1
past_range = 3
replay_size = 32

_, policy, load_net = build_networks((input_depth * past_range,) + screen, env.action_space.n)

load_net.compile(optimizer=RMSprop(clipnorm=1.), loss='mse')  # clipnorm=1.
load_net.load_weights(args.model)

observations = np.zeros((input_depth * past_range,) + screen)

def save_observation(observation):
    global observations
    observations = np.roll(observations, -input_depth, axis=0)
    observations[-input_depth:, ...] = rgb2gray(imresize(observation, screen))[None, ...]

def choose_action(observation):
    save_observation(observation)
    policy_prob = policy.predict(observations[None, ...])[0]
    policy_prob /= np.sum(policy_prob)
    return np.random.choice(np.arange(env.action_space.n), p=policy_prob)

game = 1
for _ in range(10):
    done = False
    episode_reward = 0
    noops = 0

    # init game
    observation = env.reset()
    for _ in range(past_range):
        save_observation(observation)

    # play one game
    print('Game #%8d; ' % (game,))
    while not done:
        env.render()
        action = choose_action(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if action == 0:
            noops += 1
        else:
            noops = 0
        if noops > 100:
            break
    print('Reward %4d; ' % (episode_reward,))
    game += 1

if args.evaldir:
    env.monitor.close()
