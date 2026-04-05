
class Neuron:

    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, inputs):
        total = 0.0
        for w, x in zip(self.weights, inputs):
            total += w * x
        total += self.bias
        return self.activation(total)
