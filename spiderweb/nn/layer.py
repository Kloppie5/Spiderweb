
class Layer:

    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        return [n.forward(inputs) for n in self.neurons]

    def backward(self, errors, lr):
        new_errors = [0.0] * len(self.neurons[0].weights)

        for j, neuron in enumerate(self.neurons):
            delta = errors[j] * neuron.activation_derivative(neuron.last_z)

            for i in range(len(neuron.weights)):
                new_errors[i] += neuron.weights[i] * delta
                neuron.weights[i] -= lr * delta * neuron.last_inputs[i]

            neuron.bias -= lr * delta

        return new_errors
