#!/usr/bin/env python

from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

