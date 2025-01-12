import os
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return -0.125 * x[0] * x[1] * (1 - x[0] - x[1])


def f(x1, x2):
    if x1 <= 0 or x2 <= 0:
        return 1 + abs(x1) + abs(x2)
    return -((1 - x1 - x2) * x1 * x2) / 8


def df_dx1(x1, x2):
    return -0.125 * x2 * (1 - 2 * x1 - x2)


def df_dx2(x1, x2):
    return -0.125 * x1 * (1 - x1 - 2 * x2)


def golden_section(xi, gradxi, l=0, r=5, error_accept=0.001):
    t = (-1 + math.sqrt(5)) / 2

    # 1 zingsnis
    L = r - l
    x1 = r - t * L
    value_of_function_at_x1 = f1(xi + x1 * (-gradxi))

    x2 = l + t * L
    value_of_function_at_x2 = f1(xi + x2 * (-gradxi))

    fx1 = value_of_function_at_x1
    fx2 = value_of_function_at_x2
    asb = 2
    while True:
        # 2 zingsnis
        if fx2 < fx1:
            l = x1
            L = r - l
            x1 = x2
            fx1 = fx2

            x2 = l + t * L  # apskaiciuojame naujaji desiniji taska
            fx2 = f1(xi + x2 * (-gradxi))
            asb += 1

        # 3 zingsnis
        else:
            r = x2  # atmetame desiniaja puse (x2; r]
            L = r - l
            x2 = x1  # desiniuoju tasku tampa ankstesnis kairysis taskas
            fx2 = fx1

            x1 = r - t * L  # naujasis kairysis taskas
            fx1 = f1(xi + x1 * (-gradxi))
            asb += 1

        if L < error_accept:
            return (x1 + x2) / 2, asb


def steepest_descent(f, starting_point, precision):
    iteration = 0
    function_call = 0
    steps = [starting_point]
    x1 = starting_point[0]
    x2 = starting_point[1]
    cur_value = math.inf
    prev_value = math.inf

    while (precision < abs(prev_value - cur_value)) or (prev_value == math.inf):
        grad = np.array([df_dx1(x1, x2), df_dx2(x1, x2)])
        function_call += 2

        gamma, tmp = golden_section(starting_point, grad)  # ieskome optimalaus zingsnio naudojant auksinio pjuvio metoda
        print("gamma: ", gamma)
        function_call += tmp
        # atnaujiname taska
        x1 = x1 - gamma * grad[0]
        x2 = x2 - gamma * grad[1]

        prev_value = cur_value
        cur_value = f(x1, x2)
        function_call += 1
        steps.append((x1, x2))

        iteration += 1

    return (x1, x2), iteration, function_call, steps


def configurePlot(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['left'].set_position('zero')
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)


def plot(points, filename, show=False):
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(0, 1.1, 0.001), np.arange(0, 1.1, 0.001))  # meshgrid [0, 1]
    Z = -0.125 * X * Y * (1 - X - Y)  # funkcija

    CS = ax.contour(X, Y, Z, 15, linewidths=0.3)
    ax.clabel(CS, inline=True, fontsize=9)

    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    ax.scatter(x_points, y_points, label="Optimization Path")

    configurePlot(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # plt.title(plotTitle, y=1.04)
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename)
    if show:
        plt.show()
    plt.close()


def plot3d(points):
    X, Y = np.meshgrid(np.arange(0, 1.1, 0.001), np.arange(0, 1.1, 0.001))
    Z = -0.125 * X * Y * (1 - X - Y)
    X1 = np.array([t[0] for t in points])
    Y1 = np.array([t[1] for t in points])
    Z1 = -0.125 * X1 * Y1 * (1 - X1 - Y1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, zorder=4, label='Funkcijos Grafikas')
    ax.scatter(X1, Y1, Z1, color='red', zorder=1, label='Metodo taškai')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("figures/3d_gradient_descent", dpi=1000)
    plt.show()


def main():
    matplotlib.use('TkAgg')
    starting_point = (0,0)
    # starting_point = (1, 1)
    # starting_point = (0.8, 0.8)

    epsilon = 0.001
    print(f"----------GREIČIAUSIO NUSILEIDIMO METODAS----------")
    optimal_point, iteration, function_calls, steps = steepest_descent(f, starting_point, epsilon)
    print(f"Optimalus taškas: {optimal_point}")
    print(f"Optimali funkcijos reikšmė: {f(optimal_point[0], optimal_point[1])}")
    print(f"Funkcijų iškvietimo kiekis: {function_calls}")
    print(f"Iteracijos: {iteration}")
    print(math.dist(optimal_point, (1 / 3, 1 / 3)))
    plot(steps, "steepest_descent", show=True)


if __name__ == "__main__":
    main()
