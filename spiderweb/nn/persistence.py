
import json

def save_network(network, path):
    data = []

    for layer in network.layers:
        layer_data = []

        for neuron in layer.neurons:
            layer_data.append({
                "weights": neuron.weights,
                "bias": neuron.bias
            })

        data.append(layer_data)

    with open(path, "w") as f:
        json.dump(data, f)

def load_network(path, neuron_class, layer_class, network_class, activation):
    with open(path, "r") as f:
        data = json.load(f)

    layers = []

    for layer_data in data:
        neurons = []

        for n in layer_data:
            neurons.append(
                neuron_class(
                    weights=n["weights"],
                    bias=n["bias"],
                    activation=activation
                )
            )

        layers.append(layer_class(neurons))

    return network_class(layers)
