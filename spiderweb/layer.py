
class Layer:

    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs
