from __future__ import annotations

import numpy as np


def s(x):
    return 1 / (1 + np.exp(-x))


def ds_dx(x):
    return s(x) * (1 - s(x))


class Perceptron:
    def __init__(self, bias_range: tuple[int, int] = (0, 5)):
        self.weight = np.random.random(1)[0]
        self.bias = np.random.randint(*bias_range)
        self.value = 0

    def evaluate(self, x):
        self.value = s(x * self.weight + self.bias)
        return self.value

    def d_dw(self, x: float | int):
        """
        Returns the derivative of the perceptron output at x with respect to its weight

        :param x: The point to evaluate the derivative at
        """
        return ds_dx(x * self.weight + self.bias) * x

    def d_dn(self, x: float | int):
        """
        Returns the derivative of this perceptron with respect to its input

        :param x: The point to evaluate the derivative at
        """
        return ds_dx(x * self.weight + self.bias) * self.weight

    def d_db(self, x: float | int):
        """
        Returns the derivative of the perceptron output with respect to its bias

        :param x: The point to evaluate the derivative at
        """
        return ds_dx(x * self.weight + self.bias)


class PerceptronNetwork:
    def __init__(self, layers):
        self.nodes = [Perceptron() for _ in range(layers)]
        self.n_layers = layers
        self.value = 0
        self.err = 0

    def evaluate(self, x):
        val = x
        for p in self.nodes:
            val = p.evaluate(val)
        return val

    def _backprop_node(self, x, n):
        """
        Returns the derivative of the out node with respect to node n at x

        :param x: The value to evaluate the derivative at
        :param n: The node index
        """
        self.evaluate(x)
        dP_dN = 1
        for i in range(self.n_layers - 1, n, -1):
            dP_dN *= self.nodes[i].d_dn(self.nodes[i - 1].value)
        return dP_dN

    def _backprop_weight(self, x, n):
        """
        Returns the derivative of the output node with respect to weight n

        :param x: The value to evaluate the derivative at
        :param n: The index of the weight
        :return: The derivative of the output node with respect to the weight
        """
        return self._backprop_node(x, n) * self.nodes[n].d_dw(x)

    def _backprop_bias(self, x, n):
        """
        Returns the derivative of the output node with respect to bias n

        :param x: The value to evaluate the derivative at
        :param n: The index of the bias
        :return: The derivative of the output node with respect to the bias
        """
        return self._backprop_node(x, n) * self.nodes[n].d_db(x)
