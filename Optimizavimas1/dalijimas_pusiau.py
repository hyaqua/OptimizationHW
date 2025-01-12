import sys

import numpy as np
from matplotlib import pyplot as plt

import defaults

intervalas = [0, 10]
itermax = 1000000


def dalijimas_pusiau(left, right, func, epsilon):
    steps = 0
    lefts = [0]
    rights = [10]
    mids = []
    calls = 0
    mid = (left + right) / 2
    mids.append(mid)
    dif = right - left
    for i in range(defaults.itermax):
        x1 = left + dif / 4
        x2 = right - dif / 4

        fmid = func(mid)
        fx1 = func(x1)
        fx2 = func(x2)
        calls +=3
        if fmid > fx1:
            right = mid
            mid = x1
        elif fmid > fx2:
            left = mid
            mid = x2
        else:
            left = x1
            right = x2
        dif = right - left

        steps += 1
        lefts.append(left)
        rights.append(right)
        mids.append(mid)

        if dif < epsilon:
            return steps, lefts, rights, mids, calls
    print("FAILED: MAX NUMBER OF STEPS EXCEEDED")


results = dalijimas_pusiau(intervalas[0], intervalas[1], defaults.funkcija, 0.0001)
print("Dalijimas pusiau:")
print(f"Zingsniai: {results[0]}")
for i in range(0, len(results[1])):
    print(f"Zingsnis: {i}, kaire: {results[1][i]}, desine: {results[2][i]}, vidurys: {results[3][i]}")
print(f"Rezultatas: {results[3][-1]}")
print(f"Funkcijos iskvietimai: {results[4]}")


x = np.linspace(defaults.intervalas[0], defaults.intervalas[1], 1000)
mids = np.array(results[3])

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.plot(x, defaults.funkcija(x))
bx = plt.gca()
bx.set_ylim([-3, 38])
bx.set_xlim([0, 6])
ax = plt.subplot(1, 1, 1)
ax.plot(mids, defaults.funkcija(mids), 'or', alpha=0.4)
plt.show()

