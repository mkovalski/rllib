#!/usr/bin/env python

from .agent import Agent
import math
import numpy as np

# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class RandomAgent():
    def step(self, legal_actions):
        return np.random.choice(legal_actions)

