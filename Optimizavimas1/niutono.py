import numpy as np
from matplotlib import pyplot as plt

import defaults


def niutono_metodas(x0, func, epsilon):
    h = 1e-5
    steps = 0
    mids = [x0]
    calls = 0
    def first_derivative(x):
        return (func(x + h) - func(x)) / h

    def second_derivative(x):
        return (first_derivative(x + h) - first_derivative(x)) / h

    for i in range(defaults.itermax):
        xn = x0 - first_derivative(x0) / second_derivative(x0)
        calls+=6
        steps += 1
        mids.append(xn)
        if abs(xn - x0) < epsilon:
            return steps, mids, calls
        x0 = xn


result = niutono_metodas(5, defaults.funkcija, 0.0001)
print(f"Zingsniai: {result[0]}")
for i in range(0, result[0] + 1):
    print(f"{defaults.funkcija(result[1][i])}")
print(f'F(Rezultatas): {defaults.funkcija(result[1][-1])}')
print(f"Iskvietimai: {result[2]}")
x = np.linspace(defaults.intervalas[0], defaults.intervalas[1], 1000)
lefts = np.array(result[1])

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


plt.plot(x, defaults.funkcija(x))
cx = plt.gca()
cx.set_ylim([-3, 38])
cx.set_xlim([0, 6])
ax = plt.subplot(1, 1, 1)
ax.plot(lefts, defaults.funkcija(lefts), 'or', alpha = 0.2)

plt.show()