
class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, errors, lr):
        for layer in reversed(self.layers):
            errors = layer.backward(errors, lr)
