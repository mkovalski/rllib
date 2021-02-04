import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

from .blocks import ConvLayer, ValueHead, PolicyHead, ResLayer

class AlphaZeroResnet(nn.Module):
    def __init__(self, inp_shape, output_shape, res_layer_number = 5, planes = 128,
            use_player_state = True, embedding_dict = {}):
        super(AlphaZeroResnet, self).__init__()

        self.use_player_state = use_player_state
        self.embedding_dict = embedding_dict

        self.inp_shape = inp_shape
        self.inp_planes = inp_shape[0]
        self.board_shape = inp_shape[1:]

        self.output_shape = output_shape

        self.conv = ConvLayer(self.inp_planes, planes = planes)
        self.res_layers = torch.nn.ModuleList([ ResLayer(inplanes = planes, planes = planes) for i in range(res_layer_number)])
        self.policyHead = PolicyHead(planes, self.board_shape, output_shape,
                use_player_state = use_player_state,
                embedding_dict = embedding_dict)
        self.valueHead = ValueHead(planes, self.board_shape, output_shape,
                use_player_state = use_player_state,
                embedding_dict = embedding_dict)

    def forward(self,s, player_state):
        s = self.conv(s)

        for res_layer in self.res_layers:
            s = res_layer(s)

        v = torch.tanh(self.valueHead(s, player_state = player_state))
        p = self.policyHead(s, player_state = player_state)

        return F.log_softmax(p, dim = 1).exp(), v

if __name__ == '__main__':
    state_shape = (1, 20, 20)
    output_shape = 32000

    net = AlphaZeroResnet(state_shape, output_shape)
    print(net)

    item = torch.tensor(np.random.random((4, *state_shape))).float()
    policy, value = net(item)
    print(policy.shape, value.shape)
