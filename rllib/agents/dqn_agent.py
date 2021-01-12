#!/usr/bin/env python

from .agent import Agent
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

EPS = 1e-8

class DQNAgent():
    def __init__(self,
                 model,
                 replay_buffer,
                 action_size,
                 gamma = 0.95,
                 eps_start = 0.95,
                 eps_end = 0.05,
                 eps_decay = 10000000,
                 batch_size = 128,
                 device = 'cuda'):

        self.model = model
        self.replay_buffer = replay_buffer
        self.action_size = action_size

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.RMSprop(self.model.parameters(), lr = 3e-4)
        self.target_model = copy.deepcopy(self.model)

        self.n_steps = 0
        self.running_loss = 0
        self.loss_steps = 0
        self._prev_step = None

    def load_model(self, latest_model):
        self.model.load_state_dict(latest_model)

    def get_loss(self):
        return self.running_loss / (self.loss_steps + EPS)

    def reset_loss(self):
        self.loss_steps = 0
        self.running_loss = 0

    def actions_to_mask(self, legal_actions):
        '''Use a mask of 0 here so we can easily subtract out from next_target'''
        mask = np.full(self.action_size, -np.inf)
        mask[legal_actions] = 0
        return mask

    def get_threshold(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.n_steps / self.eps_decay)

    def _get_action(self, state, legal_actions, is_eval = False):

        if is_eval or np.random.random() > self.get_threshold():
            state = torch.tensor(state).float().to(self.device)
            state = state.view((1, *state.shape))

            with torch.no_grad():
                action = self.model(state).float().cpu().numpy().flatten()
            action_idx = np.argmax(action[legal_actions])
            return legal_actions[action_idx]

        else:
            action = np.random.choice(legal_actions)
            return action

    def _get_next_target(self, next_state):
        with torch.no_grad():
            return self.target_model(next_state).cpu().numpy()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        batch = self.replay_buffer.sample(self.batch_size)
        next_state = torch.tensor(batch.next_state).float().to(self.device)

        # Get the next target for our model
        next_target = self._get_next_target(next_state)

        # Update the targets so they reflect valid actions
        # Legal actions are 0, illegal are -inf
        next_target += batch.legal_actions
        #next_target[np.where(batch.legal_actions == 0)] = float('-inf')
        next_target = np.amax(next_target, axis = 1)

        target = batch.reward + ((1 - batch.done) * (self.gamma * next_target))
        target = target.reshape(-1, 1)

        target = torch.tensor(target).float().to(self.device)

        # Clean up the original actions to see what we took
        state = torch.from_numpy(batch.state).float().to(self.device)

        pred = self.model(state).gather(
                1, torch.from_numpy(batch.action.reshape(-1, 1)).to(self.device))

        loss = F.smooth_l1_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.running_loss += loss.item()
        self.loss_steps += 1

        return loss.item()

    def _update_replay_buffer(self, state, action, legal_actions, done, reward):
        if self._prev_step is not None:
            self.replay_buffer.push(self._prev_step['state'],
                                    self._prev_step['legal_actions'],
                                    self._prev_step['action'],
                                    state,
                                    reward,
                                    done)

        if not done:
            self._prev_step = dict(state = state,
                                    legal_actions = self.actions_to_mask(legal_actions),
                                    action = action)
        else:
            self._prev_step = None

    def step(self, state, legal_actions, done, reward, is_eval = False):

        action = None
        if not done:
            action = self._get_action(state = state, legal_actions = legal_actions, is_eval = is_eval)

        if not is_eval:
            self._update_replay_buffer(state, action, legal_actions, done, reward)

        self.n_steps += 1

        return action

    def pop_transitions(self):
        pass

