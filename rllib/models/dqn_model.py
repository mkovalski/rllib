#!/usr/bin/env python

import logging
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQNModel, self).__init__()

        self.linear1 = nn.Linear(inputs, inputs // 2)
        self.linear2 = nn.Linear(inputs // 2, inputs // 4)
        self.head = nn.Linear(inputs // 4, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)
