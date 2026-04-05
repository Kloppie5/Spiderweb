from activations import step
from layer import Layer
from network import Network
from neuron import Neuron

l1 = Layer([
    Neuron([0.5, -0.5], 0.1, step),
    Neuron([-0.3, 0.3], -0.2, step)
])

l2 = Layer([
    Neuron([0.7, -0.4], 0.0, step)
])

net = Network([l1, l2])

print(net.forward([1.0, 2.0]))
