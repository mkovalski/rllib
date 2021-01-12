#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class DQNConvModel(nn.Module):

    def __init__(self, inputs, outputs, kernel_size = 3):
        super(DQNConvModel, self).__init__()

        planes = inputs[0]
        x = inputs[1]
        y = inputs[2]


        self.conv1 = nn.Conv2d(planes, 128, kernel_size = kernel_size, stride = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, kernel_size = kernel_size, stride = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, kernel_size = kernel_size, stride = 1,
                               bias = False)
        self.bn3 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)

        self.output_shape = self._get_output_shape(inputs)

        self.linear1 = nn.Linear(self.output_shape, self.output_shape)

        self.final = nn.Linear(self.output_shape, outputs)

    def _get_output_shape(self, sh):
        with torch.no_grad():
            out = torch.rand((1, *sh))

            out = self.bn1(self.conv1(out))
            out = self.maxpool(out)
            out = self.bn2(self.conv2(out))
            out = self.maxpool(out)
            out = self.bn3(self.conv3(out))
            out = self.maxpool(out)

        return np.prod(out.shape)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.linear1(x))

        return self.final(x)

if __name__ == '__main__':
    inp_shape = (1, 20, 20)
    output_shape = 100

    model = DQNConvModel(inp_shape, output_shape, kernel_size = 3)
    print(model)

    data = torch.rand((1, *inp_shape))
    print("Input shape: {}".format(data.shape))

    out = model(data)
    print("Output shape: {}".format(out.shape))




