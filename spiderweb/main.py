from neuron import Neuron
from activations import step

neuron = Neuron(
    weights = [0.5, -0.5],
    bias = 0.1,
    activation = step
)

inputs = [1.0, 2.0]

print("Output:", neuron.forward(inputs))
