import math


def function(x):
    return (
        5 * pow(x, 4)
        - 22.4 * pow(x, 3)
        + 15.85272 * pow(x, 2)
        + 24.161472 * x
        - 23.4824832
    )


def exactFirstDerivative(x):
    return 20 * pow(x, 3) - 67.2 * pow(x, 2) + 31.70544 * x + 24.161472


def exactSecondDerivative(x):
    return 60 * pow(x, 2) - 134.4 * x + 31.70544


def exactDerivative(x):
    variable = x
    while True:
        result = exactFirstDerivative(variable)
        if result <= 0.0001 and result >= -0.0001:
            return variable

        variable = variable - exactFirstDerivative(variable) / exactSecondDerivative(
            variable
        )


def approximationFirstDerivative(x):
    return (function(x + 0.01) - function(x)) / 0.01


def approximationSecondDerivative(x):
    return (function(x + 0.01) - 2 * function(x) + function(x - 0.01)) / 0.0001


def approximationDerivative(x):
    variable = x
    while True:
        result = approximationFirstDerivative(variable)
        if result <= 0.0001 and result >= -0.0001:
            return variable
        variable = variable - approximationFirstDerivative(
            variable
        ) / approximationSecondDerivative(variable)


def main():
    print("Using ExactDerivative")
    print("Find using exactDerivative with -1 : " + str(exactDerivative(-1)))
    print("Find using exactDerivative with 2 : " + str(exactDerivative(2)))
    print("")
    print("Using Approximation")
    print(
        "Find using approximateDerivative with -1 : " + str(approximationDerivative(-1))
    )
    print(
        "Find using approximateDerivative with 2 : " + str(approximationDerivative(2))
    )


if __name__ == "__main__":
    main()
