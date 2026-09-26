import math

def sigmoid(x):
    if isinstance(x, (list, tuple)):
        return [1.0 / (1.0 + math.exp(-element)) for element in x]
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(sigmoid_output):
    if isinstance(sigmoid_output, (list, tuple)):
        return [s * (1.0 - s) for s in sigmoid_output]

    return sigmoid_output * (1.0 - sigmoid_output)
def relu(x):
    return max(x, 0)
def tanh(x):
    if isinstance(x, (list, tuple)):
        return [math.tanh(element) for element in x]
    return math.tanh(x)


def tanh_derivative(tanh_output):
    if isinstance(tanh_output, (list, tuple)):
        return [1.0 - t * t for t in tanh_output]
    return 1.0 - tanh_output * tanh_output
def neuron(inputs, weights, bias):
    if len(inputs) != len(weights):
        raise ValueError("inputs and weights must have the same length")

    return sum(x * w for x, w in zip(inputs, weights)) + bias
def mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("inputs must have the same length")

    return sum(
        (a - p) ** 2
        for a, p in zip(actual, predicted)
    ) / len(actual)