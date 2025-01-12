import math
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    if x1 <= 0 or x2 <= 0:
        return 1 + abs(x1) + abs(x2)
    return -((1 - x1 - x2) * x1 * x2) / 8


def df_dx1(x1, x2):
    return -0.125 * x2 * (1 - 2 * x1 - x2)


def df_dx2(x1, x2):
    return -0.125 * x1 * (1 - x1 - 2 * x2)


def gradient_descent(starting_point, gamma, epsilon):
    steps = [starting_point]
    max_iterations = 1000
    iteration = 0
    function_calls = 0
    xi = list(starting_point)
    while iteration < max_iterations:
        grad_numeric = [df_dx1(xi[0], xi[1]), df_dx2(xi[0], xi[1])]
        function_calls += 2

        xi_new = [xi[j] - gamma * grad_numeric[j] for j in range(2)]
        steps.append(xi_new)

        iteration += 1
        grad_norm = np.linalg.norm(np.array(grad_numeric) * gamma)
        if grad_norm < epsilon:
            break

        xi = xi_new

    return xi, iteration, function_calls, steps


def plot(points, filename, show=False):
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(0, 1.1, 0.001), np.arange(0, 1.1, 0.001))
    Z = -0.125 * X * Y * (1 - X - Y)

    CS = ax.contour(X, Y, Z, 40, linewidths=0.5)
    ax.clabel(CS, inline=True, fontsize=9)

    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    ax.scatter(x_points, y_points, label="Optimization Path")
    ax.scatter(x_points[-1],y_points[-1], label="Result", color="red")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename)
    if show:
        plt.show()
    plt.close()


def plot3d(points, filename):
    X, Y = np.meshgrid(np.arange(0, 1.1, 0.001), np.arange(0, 1.1, 0.001))
    Z = -0.125 * X * Y * (1 - X - Y)
    X1 = np.array([t[0] for t in points])
    Y1 = np.array([t[1] for t in points])
    Z1 = -0.125 * X1 * Y1 * (1 - X1 - Y1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, zorder=4, label='Funkcijos Grafikas')
    ax.scatter(X1, Y1, Z1, color='red', zorder=1, label='Metodo taškai')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/3d_" + filename)
    plt.show()


def main():
    matplotlib.use('TkAgg')
    # starting_point = (0, 0)
    # starting_point = (1, 1)
    starting_point = (0.8, 0.8)

    gradientas = [df_dx1(starting_point[0], starting_point[1]), df_dx2(starting_point[0], starting_point[1])]
    print(gradientas)
    print(f(starting_point[0], starting_point[1]))
    gamma = 3
    epsilon = 0.001

    print(f"----------GRADIENTINIO NUSILEIDIMO METODAS----------")
    optimal_point, iteration, function_calls, steps = gradient_descent(starting_point, gamma, epsilon)
    print(f"Optimalus taškas: {optimal_point}")
    print(f"Optimali funkcijos reikšmė: {f(optimal_point[0], optimal_point[1])}")
    print(f"Funkcijų iškvietimo kiekis: {function_calls}")
    print(f"Iteracijos: {iteration}")
    print(math.dist(optimal_point, (1 / 3, 1 / 3)))

    plot(steps, "gradient_descent", show=True)
    # plot3d(steps, "gradient_descent")


if __name__ == "__main__":
    main()
