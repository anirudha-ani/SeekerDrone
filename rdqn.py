'''
Author: Sunghoon Hong
Title: RDQNAgent.py
Descr
    Recurrent Deep Q-Network Agent for Airsim,
Detail:
    - not use join()
    - reset for zero-image error
    - tensorflow v1 + keras
    - hard update for target model

'''


import os
import csv
import time
import random
import argparse
from collections import deque
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Lambda, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from tensorflow.keras.optimizers import Adam
from PIL import Image
import cv2
from airsim_env import Env#, ACTION

np.set_printoptions(suppress=True, precision=4)
agent_name = 'rdqn'


class RDQNAgent(tf.keras.Model):
    
    def __init__(self, state_size, action_size, lr,
                gamma, batch_size, memory_size, 
                epsilon, epsilon_end, decay_step, load_model):
        super(RDQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = Adam(learning_rate=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.critic1 = Dense(128, kernel_initializer='he_normal', use_bias=False, input_shape=[24,])
        self.critic2 = Dense(128, kernel_initializer='he_normal', use_bias=False)
        self.critic3 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.elu1 = ELU()
        self.elu2 = ELU()
        

        self.memory = deque(maxlen=self.memory_size)

    @tf.function
    def call(self, state):
        state = tf.reshape(state, [-1, 352])
        # Critic
        Qvalue = self.critic1(state)
        Qvalue = self.norm1(Qvalue)
        Qvalue = self.elu1(Qvalue)
        Qvalue = self.critic2(Qvalue)
        Qvalue = self.norm2(Qvalue)
        Qvalue = self.elu2(Qvalue)
        Qvalue = self.critic3(Qvalue)

        return Qvalue


    def loss_function(self, pred, actions, targets):
        action_vec = tf.one_hot(action, self.action_size)
        Q = tf.reduce_sum(pred * action_vec, axis=1)
        error = tf.abs(targets - Q)
        quadratic = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - quadratic
        loss = tf.reduce_mean(0.5 * tf.math.square(quadratic) + linear)
        return loss

    def get_action(self, state, train):
        Qs = self.call(state.reshape(1,-1))
        Qmax = np.amax(Qs)
        if train and np.random.random() < self.epsilon:
            return np.random.choice(self.action_size), np.argmax(Qs), Qmax
        return np.argmax(Qs), np.argmax(Qs), Qmax


    def append_memory(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))



def train_model(agent, agent_target):
    batch = random.sample(agent.memory, agent.batch_size)

    images = np.zeros([agent.batch_size] + agent.state_size)
    actions = np.zeros((agent.batch_size))
    rewards = np.zeros((agent.batch_size))
    next_images = np.zeros([agent.batch_size] + agent.state_size)
    dones = np.zeros((agent.batch_size))

    targets = np.zeros((agent.batch_size, 1))

    for i, sample in enumerate(batch):
        images[i] = sample[0]
        actions[i] = sample[1]
        rewards[i] = sample[2]
        next_images[i] = sample[3]
        dones[i] = sample[4]
    states = images
    next_states = next_images

    with tf.GradientTape() as tape:
        target_next_Qs = agent_target.call(next_states)
        targets = rewards + agent.gamma * (1 - dones) * np.amax(target_next_Qs, axis=1)
        pred = agent.call(states)
        loss = agent.loss_function(pred, actions, targets)
    grads = tape.gradient(loss, agent.trainable_variables)
    agent.optimizer.apply_gradients(zip(grads, agent.trainable_variables))

    return loss


def update_target_model(agent, agent_target):
    agent_target.set_weights(agent.get_weights())
        
def load_model(name):
    model = None
    if os.path.exists(name):
        model = keras.models.load_model(name)
        print('Model loaded')
    else:
        print("loaded failed")
    return model

def save_model(agent, name):
    agent.save(name)

    


'''
Environment interaction
'''



if __name__ == '__main__':
    # CUDA config
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play',       action='store_true')
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seqsize',    type=int,   default=8)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--memory_size',type=int,   default=50000)
    # parser.add_argument('--train_start',type=int,   default=600)
    parser.add_argument('--train_start',type=int,   default=50)
    parser.add_argument('--train_rate', type=int,   default=5)
    parser.add_argument('--target_rate',type=int,   default=1000)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.05)
    parser.add_argument('--decay_step', type=int,   default=20000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name):
        os.makedirs('save_graph/'+ agent_name)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent
    # state_size = [args.seqsize, args.img_height, args.img_width, 1]
    # size of outs
    state_size = [args.seqsize, 44]
    # state_size = [24]
    action_size = 3
    agent = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )

    agent.compute_output_shape(input_shape=(None, args.seqsize, 44))
    agent_target = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )
    agent_target.compute_output_shape(input_shape=(None, args.seqsize, 44))

    if args.load_model:
        load_model('./save_model/'+ agent_name)
    update_target_model(agent, agent_target)


    episode = 0
    env = Env()

    if args.play:
        while True:
            try:
                goal = np.zeros([4,6])
                goal[0][1] = 1
                goal[1][2] = 1
                goal[1][3] = 1
                goal[2][2] = 1  #2 behind
                goal[3][3] = 1  #3 right
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgQ = 0., 0, 0., 0.

                observe = env.reset()
                state = np.append(goal, observe, 1)
                state = np.reshape(state, [-1])
                history = np.tile(state,[args.seqsize,1,1])
                state = history
                
                while not done:
                    timestep += 1
                    action, policy, Qmax = agent.get_action(state, False)
                    observe, reward, done, info = env.step(action, goal)

                    next_state = np.append(goal, observe, 1)
                    next_state = np.reshape(next_state, [-1])
                    history = np.append(history[1:, :], next_state,axis=0)
                    next_state = history

                    # stats
                    avgQ += float(Qmax)
                    score += reward

                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                if bug:
                    continue
                
                avgQ /= timestep

                # done
                print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f Info %s'
                        % (episode, bestY, timestep, score, avgQ, info['status']))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        time_limit = 60
        highscoreY = 0.
        if os.path.exists('save_stat/'+ agent_name + '_stat.csv'):
            with open('save_stat/'+ agent_name + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name + '_highscore.csv'):
            with open('save_stat/'+ agent_name + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)
        global_step = 0
        global_train_num = 0

        while True:
            try:
                
                goal = np.zeros([4,6])
                goal[0][1] = 1
                goal[1][2] = 1
                goal[1][3] = 1
                goal[2][2] = 1  #2 behind
                goal[3][3] = 1  #3 right

                done = False
                bug = False

                # stats
                timestep, score, avgQ = 0, 0., 0.
                train_num, loss = 0, 0.

                observe = env.reset()

                state = np.append(goal, observe, 1)
                state = np.reshape(state, [-1])
                history = np.tile(state,[args.seqsize,1])
                state = history
                while not done and timestep < time_limit:
                    timestep += 1
                    global_step += 1
                    if len(agent.memory) >= args.train_start and global_step >= args.train_rate:
                        for _ in range(args.epoch):
                            c_loss = train_model(agent, agent_target)
                            loss += float(c_loss)
                            train_num += 1
                            global_train_num += 1
                        global_step = 0
                    if global_train_num >= args.target_rate:
                        update_target_model(agent, agent_target)
                        global_train_num = 0

                    action, policy, Qmax = agent.get_action(state, True)
                    observe, reward, done, info = env.step(action, goal)

                    if timestep % 5:
                        print(reward, observe[:,-1])
                    
                    # history = np.append(history[:, 1:], [observe], axis=1)

                    # next_state = [history]
                    next_state = np.append(goal, observe, 1)
                    next_state = np.reshape(next_state, [1,-1])
                    history = np.append(history[1:, :], next_state,axis=0)
                    next_state = history
                    agent.append_memory(state, action, reward, next_state, done)

                    # stats
                    avgQ += float(Qmax)
                    score += reward

                    if args.verbose:
                        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, real_action, reward, info['status']))

                    state = next_state

                    if agent.epsilon > agent.epsilon_end:
                        agent.epsilon -= agent.epsilon_decay

                if bug:
                    continue
                if train_num:
                    loss /= train_num
                avgQ /= timestep

                stats = [
                    episode, timestep, score, \
                    loss, avgQ
                ]
                # log stats
                with open('save_stat/'+ agent_name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])
                
                print(episode, score, avgQ, loss)
                save_model(agent, './save_model/'+ agent_name)
                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break