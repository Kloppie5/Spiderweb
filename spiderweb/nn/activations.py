
def step(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + (2.71828 ** -x))  # no math.exp yet

def relu(x):
    return x if x > 0 else 0
