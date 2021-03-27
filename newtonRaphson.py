import math


def function(x):
    return (
        5 * pow(x, 4)
        - 22.4 * pow(x, 3)
        + 15.85272 * pow(x, 2)
        + 24.161472 * x
        - 23.4824832
    )


def derivative(x):
    return 20 * pow(x, 3) - 67.2 * pow(x, 2) + 31.70544 * x + 24.161472


def newton(a):
    x = a
    if function(x) <= 0.0001 and function(x) >= -0.0001:
        return x
    while True:
        if derivative(x) == 0:
            return -10000
        x = x - function(x) / derivative(x)
        if function(x) <= 0.0001 and function(x) >= -0.0001:
            return x


print(derivative(1.2))
print(newton(-1.1))
print(newton(1.1))
print(newton(1.3))
print(newton(3))
