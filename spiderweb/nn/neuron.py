
class Neuron:

    def __init__(self, weights, bias, activation, activation_derivative):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.activation_derivative = activation_derivative

        self.last_inputs = None
        self.last_z = None
        self.last_output = None

    def forward(self, inputs):
        self.last_inputs = inputs

        z = 0.0
        for w, x in zip(self.weights, inputs):
            z += w * x
        z += self.bias

        self.last_z = z
        self.last_output = self.activation(z)
        return self.last_output
