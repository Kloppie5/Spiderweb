from nn.activations import step
from nn.layer import Layer
from nn.network import Network
from nn.neuron import Neuron

from visualizations.network_plot import print_network

l1 = Layer([
    Neuron([0.5, -0.5], 0.1, step),
    Neuron([-0.3, 0.3], -0.2, step)
])

l2 = Layer([
    Neuron([0.7, -0.4], 0.0, step)
])

net = Network([l1, l2])

print_network(net)

print(net.forward([1.0, 2.0]))
