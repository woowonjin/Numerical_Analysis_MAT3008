import math


def function(x):
    return (
        5 * pow(x, 4)
        - 22.4 * pow(x, 3)
        + 15.85272 * pow(x, 2)
        + 24.161472 * x
        - 23.4824832
    )


def bisection(a, b):
    first = a
    second = b
    if function(a) <= 0.0001 and function(a) >= -0.0001:
        return a
    if function(b) <= 0.0001 and function(b) >= -0.0001:
        return b
    while True:
        c = (first + second) / 2
        equation = function(c)
        if equation <= 0.0001 and equation >= -0.0001:
            return c
        elif equation > 0:
            if function(a) > 0:
                first = c
            else:
                second = c
        else:
            if function(a) > 0:
                second = c
            else:
                first = c


print(bisection(-1.5, -1))
print(bisection(1, 1.2))
print(bisection(3.1, 3.2))

