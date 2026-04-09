
import random

class Trainer:

    def __init__(self, network):
        self.network = network

    def get_result(self, img):
        return self.network.forward(img)

    def train_from_result(self, result, label, lr=0.1):
        target = [0.0] * 10
        target[label] = 1.0

        errors = [result[i] - target[i] for i in range(10)]

        self.network.backward(errors, lr)
