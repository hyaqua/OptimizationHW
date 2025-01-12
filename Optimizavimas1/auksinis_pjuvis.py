import numpy as np
from matplotlib import pyplot as plt

import defaults



def auksinis_pjuvis(l, r, func, epsilon):
    steps = 0
    lefts = []
    rights = []
    calls = 0
    tau = ((-1 + 5 ** 0.5) / 2)
    dif = r - l
    x1 = r - (tau * dif)
    x2 = l + (tau * dif)
    fx1 = func(x1)
    fx2 = func(x2)
    calls+=2
    for i in range(defaults.itermax):
        if fx2 < fx1:
            l = x1
            x1 = x2
            fx1 = fx2
            dif = r - l
            x2 = l + tau * dif
            fx2 = func(x2)
        else:
            r = x2
            x2 = x1
            fx2 = fx1
            dif = r - l
            x1 = r - tau * dif
            fx1 = func(x1)
        calls += 1

        steps += 1
        lefts.append(l)
        rights.append(r)

        if dif < epsilon:
            print(dif)
            return steps, lefts, rights,calls


result = auksinis_pjuvis(0, 10, defaults.funkcija, 0.0001)
print("Dalijimas pusiau:")
print(f"Zingsniai: {result[0]}")
for i in range(0, len(result[1])):
    # print(f"Zingsnis: {i}, kaire: {result[1][i]}, desine: {result[2][i]}")
    print(f"{result[2][i]}")
print(f"Rezultatas: {result[1][-1]}, {result[2][-1]}")
print(f"F(Rezultatas): {defaults.funkcija(result[1][-1])}, {defaults.funkcija(result[2][-1])}")

print(f"Iskvietimai: {result[3]}")
x = np.linspace(defaults.intervalas[0], defaults.intervalas[1], 1000)
lefts = np.array(result[1])
rights = np.array(result[2])

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


plt.plot(x, defaults.funkcija(x))
cx = plt.gca()
cx.set_ylim([-3, 38])
cx.set_xlim([0, 6])
ax = plt.subplot(1, 1, 1)
ax.plot(lefts, defaults.funkcija(lefts), 'or', alpha = 0.2)
bx = plt.subplot(1, 1, 1)
bx.plot(rights, defaults.funkcija(rights), 'og', alpha = 0.2)

plt.show()