import numpy as np
import matplotlib.pyplot as plt
import pytest
import tqdm

import network


@pytest.fixture(autouse=True)
def seed_np():
    np.random.seed(1)


def test_perceptron_initialization():
    p = network.Perceptron()

    assert 0 < p.weight < 1
    assert isinstance(p.weight, float)

    assert 0 < p.bias < 5
    assert isinstance(p.bias, int)


def test_perceptron_evaluation():
    p = network.Perceptron()
    p.weight = 1
    p.bias = 0.25

    assert p.evaluate(0.25) == network.s(0.5)


def test_perceptron_backprop():
    p = network.Perceptron()
    points = np.array([[0, 1], [1, 0]])
    learning_rate = 0.01

    err = []

    for i in range(10000):
        np.random.shuffle(points)

        mse = 0
        for X, Y in points:
            val = p.evaluate(X)
            mse += (Y - val)**2 / len(points)
            dp_dw = -2 * (Y - val) * p.d_dw(X)
            dp_db = -2 * (Y - val) * p.d_db(X)

            p.weight -= learning_rate * dp_dw
            p.bias -= learning_rate * dp_db

        err.append(mse)

    plt.plot(np.arange(len(err)), err)
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.suptitle("Perceptron Training Test")
    plt.title(f"Start Err: {err[0]:0.5f} ; End Err: {err[-1]:0.5f}", fontsize=10)

    plt.show()

    assert err[0] > err[-1]


def test_network_ff():
    nn = network.PerceptronNetwork(2)

    assert nn.evaluate(1) == 0.8722984271655249


def C(x, y):
    """
    Returns the cost function defined as (Y-X)^2
    where X is the predicted value and Y is the real value.

    :param x: The predicted value
    :param y: The real value
    """
    return (y - x)**2


def dC_dx(x, y):
    """
    Returns the derivative of the cost function C with respect to x

    :param x: The predicted value to evaluate the cost function at
    :param y: The real value
    """
    return -2*(y - x)


def test_network_backprop():
    np.random.seed(0)
    nn = network.PerceptronNetwork(4)

    train_data = np.array([[1, 0], [0, 1]], dtype=np.float64)
    learning_rate = 0.01

    err = []
    for i in (trange := tqdm.trange(10000)):
        np.random.shuffle(train_data)
        mse = 0
        for X, Y in train_data:
            layer_err = 0
            for n, p in enumerate(nn.nodes):
                val = nn.evaluate(X)
                layer_err = C(val, Y)
                dP_dwn = dC_dx(val, Y) * nn._backprop_weight(X, n)
                dP_dbn = dC_dx(val, Y) * nn._backprop_bias(X, n)

                p.weight -= learning_rate * dP_dwn
                p.bias -= learning_rate * dP_dbn
            mse += layer_err / len(train_data)
        err.append(mse)

        if i % 100 == 0:
            trange.set_description(f"MSE: {mse:0.5f}")

    plt.plot(np.arange(len(err)), err)
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.suptitle("Perceptron Training Test")
    plt.title(f"Start Err: {err[0]:0.5f} ; End Err: {err[-1]:0.5f}", fontsize=10)

    plt.show()

    assert err[0] > err[-1]
