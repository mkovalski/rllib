#!/usr/bin/env python

'''Simple replay buffer for reinforcement learning tasks'''

from collections import namedtuple
import numpy as np
import pickle
import random
from tqdm import tqdm

# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
transition_items = ['state', 'legal_actions', 'action', 'next_state', 'reward', 'done']

Transition = namedtuple('Transition', tuple(transition_items))

class ReplayBuffer():
    '''Simple replay buffer for reinfocement learning

    Args:
        capacity (int): Size of replay buffer

    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''Saves transition'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''Randomly sample a batch from the replay buffer'''
        samples = random.sample(self.memory, batch_size)

        # Aggregate numpy arrays
        args = {}

        for idx, field in enumerate(samples[0]._fields):
            args[field] = np.stack([samples[i][idx] for i in range(len(samples))])

        new_sample = Transition(**args)

        return new_sample

    def populate(self, env):
        state = env.reset()

        for i in tqdm(range(self.capacity)):
            action = env.sample()
            next_state, reward, done, _ = env.step(action)
            self.push(state, action, next_state, reward, done)
            state = next_state

            if done:
                state = env.reset()

    def pop_all(self):
        self.position = 0
        return_list = []
        return_list, self.memory = self.memory, return_list
        return return_list

    def save(self, path):
        with open(path, 'wb') as myFile:
            pickle.dump(self, myFile)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as myFile:
            rb = pickle.load(myFile)
        return rb

    def __len__(self):
        return len(self.memory)
