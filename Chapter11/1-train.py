from __future__ import print_function
from __future__ import division
from scipy.misc import imresize
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import time
import os
import gym
import numpy as np
import h5py
import argparse
from model import build_networks
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='Breakout-v0', help='OpenAI gym environment name', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent', type=int)
parser.add_argument('--learning_rate', default=0.001, help='Learning rate', type=float)
parser.add_argument('--steps', default=80000000, help='Number of frames to decay learning rate', type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', type=int)
parser.add_argument('--save_freq', default=250000, help='Number of frames before saving weights', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', type=int)
parser.add_argument('--reward_scale', default=1., type=float)
parser.add_argument('--beta', default=0.01, type=float)
args = parser.parse_args()

screen=(84, 84)
input_depth = 1
past_range = 3
observation_shape = (input_depth * past_range,) + screen
discount=0.99

def policy_loss(advantage=0., beta=0.01):
    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))
    return loss

def value_loss():
    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))
    return loss

def learn_proc(mem_queue, weight_dict):
    try:
        pid = os.getpid()
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=False,lib.cnmem=0.3,compiledir=th_comp_learn'

        print(' %5d> Learning process' % (pid,))

        env = gym.make(args.game)

        _, _, train_network = build_networks(observation_shape, env.action_space.n)

        advantage = Input(shape=(1,))
        train_net = Model(inputs=train_network.inputs + [advantage], outputs=train_network.outputs)
        train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99), loss=[value_loss(), policy_loss(advantage, args.beta)])

        pol_loss = deque(maxlen=25)
        val_loss = deque(maxlen=25)
        values = deque(maxlen=25)
        entropy = deque(maxlen=25)
        swap_counter = args.swap_freq
        unroll = np.arange(args.batch_size)
        targets = np.zeros((args.batch_size, env.action_space.n))

        if args.checkpoint > 0:
            print(' %5d> Loading weights from file' % (pid,))
            train_net.load_weights('model-%s-%d.h5' % (args.game, args.checkpoint,))

        print(' %5d> Setting weights in dict' % (pid,))
        weight_dict['update'] = 0
        weight_dict['weights'] = train_net.get_weights()

        last_obs = np.zeros((args.batch_size,) + observation_shape)
        actions = np.zeros(args.batch_size, dtype=np.int32)
        rewards = np.zeros(args.batch_size)

        idx = 0
        counter = args.checkpoint
        save_counter = args.checkpoint % args.save_freq + args.save_freq
        while True:
            last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
            idx = (idx + 1) % args.batch_size
            if idx == 0:
                learning_rate = max(0.00000001, (args.steps - counter) / args.steps * args.learning_rate)

                K.set_value(train_net.optimizer.lr, learning_rate)
                frames = len(last_obs)
                counter += frames

                values_, policy = train_net.predict([last_obs, unroll])

                targets.fill(0.)
                advantage = rewards - values_.flatten()
                targets[unroll, actions] = 1.

                loss = train_net.train_on_batch([last_obs, advantage], [rewards, targets])
                entropy_ = np.mean(-policy * np.log(policy + 0.00000001))
                pol_loss.append(loss[2])
                val_loss.append(loss[1])
                entropy.append(entropy_)
                values.append(np.mean(values_))
                min_val, max_val, avg_val = min(values), max(values), np.mean(values)
                print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
                      '--- Value-Loss: %10.6f; Avg: %10.6f '
                      '--- Entropy: %7.6f; Avg: %7.6f '
                      '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                          counter,
                          loss[2], np.mean(pol_loss),
                          loss[1], np.mean(val_loss),
                          entropy_, np.mean(entropy),
                          min_val, max_val, avg_val), end='')

                swap_counter -= frames
                if swap_counter < 0:
                    swap_counter += args.swap_freq
                    weight_dict['weights'] = train_net.get_weights()
                    weight_dict['update'] += 1

            save_counter -= 1
            if save_counter < 0:
                save_counter += args.save_freq
                train_net.save_weights('model-%s-%d.h5' % (args.game, counter,), overwrite=True)

    except Exception as e:
        print(e)



def generate_experience_proc(mem_queue, weight_dict, no):
    try:
        pid = os.getpid()
        os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu,nvcc.fastmath=True,lib.cnmem=0,compiledir=th_comp_act_' + str(no)
        print(' %5d> Process started' % (pid,))

        frames = 0

        env = gym.make(args.game)

        value_net, policy_net, load_net = build_networks(observation_shape, env.action_space.n)

        value_net.compile(optimizer='rmsprop', loss='mse')
        policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.])  # dummy loss

        observations = np.zeros(observation_shape)
        last_observations = np.zeros_like(observations)

        n_step_observations = deque(maxlen=args.n_step)
        n_step_actions = deque(maxlen=args.n_step)
        n_step_rewards = deque(maxlen=args.n_step)

        while 'weights' not in weight_dict:
            time.sleep(0.1)
        load_net.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

        best_score = 0
        avg_score = deque([0], maxlen=25)

        counter = 0
        last_update = 0
        while True:
            done = False
            episode_reward = 0
            op_last, op_count = 0, 0
            observation = env.reset()
            for _ in range(past_range):
                last_observations = observations[...]
                observations = np.roll(observations, -input_depth, axis=0)
                observations[-input_depth:, ...] = rgb2gray(imresize(observation, screen))[None, ...]

            while not done:
                frames += 1

                # choose action
                policy = policy_net.predict(observations[None, ...])[0]
                action = np.random.choice(np.arange(env.action_space.n), p=policy)

                observation, reward, done, _ = env.step(action)
                episode_reward += reward
                best_score = max(best_score, episode_reward)

                # save observations
                last_observations = observations[...]
                observations = np.roll(observations, -input_depth, axis=0)
                observations[-input_depth:, ...] = rgb2gray(imresize(observation, screen))[None, ...]
                reward = np.clip(reward, -1., 1.)

                n_step_observations.appendleft(last_observations)
                n_step_actions.appendleft(action)
                n_step_rewards.appendleft(reward)
                counter += 1
                if done or counter >= args.n_step:
                    r = 0.
                    if not done:
                        r = value_net.predict(observations[None, ...])[0]
                    for i in range(counter):
                        r = n_step_rewards[i] + discount * r
                        mem_queue.put((n_step_observations[i], n_step_actions[i], r))
                    counter = 0
                    n_step_observations.clear()
                    n_step_actions.clear()
                    n_step_rewards.clear()

                op_count = 0 if op_last != action else op_count + 1
                done = done or op_count >= 100
                op_last = action

                if frames % 2000 == 0:
                    print(' %5d> Best: %4d; Avg: %6.2f; Max: %4d' % (
                        pid, best_score, np.mean(avg_score), np.max(avg_score)))
                if frames % args.batch_size == 0:
                    update = weight_dict.get('update', 0)
                    if update > last_update:
                        last_update = update
                        load_net.set_weights(weight_dict['weights'])

            avg_score.append(episode_reward)

    except Exception as e:
        print(e)


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)

    pool = Pool(args.processes + 1, init_worker)

    try:
        for i in range(args.processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, i))

        pool.apply_async(learn_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
