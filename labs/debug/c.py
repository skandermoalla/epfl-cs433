from debug.a import stochastic_gradient
from debug.b import gradient1


def c(x):
    return stochastic_gradient(x)


gradient = gradient1
print(c(0))