
def print_network(network):
    
    for i, layer in enumerate(network.layers):
        print(f"Layer {i}: {len(layer.neurons)} neurons")

        for j, neuron in enumerate(layer.neurons):
            print(f"  Neuron {j}: weights={neuron.weights}, bias={neuron.bias}")
