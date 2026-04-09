
def step(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return x if x > 0 else 0
