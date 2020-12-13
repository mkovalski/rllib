#!/usr/bin/env python

from .agent import Agent
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent():
    def __init__(self,
                 model,
                 env,
                 replay_buffer):

        self.model = model
        self.env = env
        self.replay_buffer = replay_buffer

        self.optimizer = optim.RMSprop(self.model.parameters(), lr = 0.0005)

    def get_action(self, threshold, state):
        if np.random.random() > threshold:
            state = torch.from_numpy(state).float().cuda()
            state = state.view(1, -1)

            with torch.no_grad():
                return self.model(state).float().cpu().numpy().flatten()
        else:
            action = self.env.sample(state)
            return action

    def optimize(self, batch_size, gamma):
        if len(self.replay_buffer) < batch_size:
            return 0

        batch = self.replay_buffer.sample(batch_size)
        next_state = torch.from_numpy(batch.next_state).float().to(DEVICE)

        with torch.no_grad():
            next_target = self.model(next_state).cpu().numpy()

        # Update the targets so they reflect valid actions
        self.env.clean_action(batch.next_state, next_target)
        next_target = np.amax(next_target, axis = 1)

        target = batch.reward + ((1 - batch.done) * (gamma * next_target))
        target = target.reshape(-1, 1)

        target = torch.from_numpy(target).float().to(DEVICE)

        # Clean up the original actions to see what we took
        indices = np.argmax(batch.action, axis = 1)
        state = torch.from_numpy(batch.state).float().to(DEVICE)

        pred = self.model(state).gather(
                1, torch.from_numpy(indices.reshape(-1, 1)).to(DEVICE))

        loss = F.smooth_l1_loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self,
              episodes = 100000,
              batch_size = 128,
              gamma = 0.999,
              eps_start = 0.95,
              eps_end = 0.05,
              eps_decay = 20000,
              evaluate_iters = 1000):

        nsteps = 0
        running_loss = 0.

        for i in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            running_loss = 0

            while not done:
                threshold = eps_end + (eps_start - eps_end) * \
                        math.exp(-1. * nsteps / eps_decay)
                action = self.get_action(threshold = threshold,
                        state = state)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.push(state,
                                        action,
                                        next_state,
                                        reward,
                                        done)

                state = next_state

                loss = self.optimize(batch_size, gamma)
                running_loss += loss
                nsteps += 1

            if i % 100 == 0:
                print("Iteration {}: Loss: {}, threshold: {}".format(i, running_loss / 100, threshold))

            if i % evaluate_iters == 0:
                self.evaluate()
                #self.print_stats()
                #self.save()

    def evaluate(self):
        done = False
        state = self.env.reset()
        move_num = 0


        while not done:
            with torch.no_grad():
                device_state = torch.from_numpy(state).float().to(DEVICE).view(1, *state.shape)
                action = self.model(device_state).cpu().numpy()

            state = state.reshape((1, *state.shape))
            self.env.clean_action(state, action)
            action = action.flatten()

            self.env.print_action(action, move_num)
            move_num += 1

            state, reward, done, _ = self.env.step(action)

        print(reward)
        print()


    def print_stats(self):
        pass

    def save(self):
        pass



