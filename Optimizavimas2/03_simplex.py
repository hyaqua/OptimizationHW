import os
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def f1(x):
    return -0.125 * x[0] * x[1] * (1 - x[0] - x[1])


def f(x1, x2):
    if x1 <= 0 or x2 <= 0:
        return 1 + abs(x1) + abs(x2)
    return -((1 - x1 - x2) * x1 * x2) / 8


def generate_simplex_method_points(starting_point, alpha=0.3):
    x0 = np.array(starting_point)
    n = len(x0)
    simplex_points = [x0]

    delta1 = ((math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2))) * alpha
    delta2 = ((math.sqrt(n + 1) - 1) / (n * math.sqrt(2))) * alpha

    for i in range(n):
        delta = []
        for j in range(n):
            if i == j:
                delta.append(delta2)
            else:
                delta.append(delta1)
        simplex_points.append(x0 + delta)
    return simplex_points


def evaluate_simplex(f, simplex_points):
    return [f(point[0], point[1]) for point in simplex_points]


def find_best_index(simplex_values):
    best_index = np.argmin(simplex_values)
    return best_index


def find_worst_index(simplex_values):
    worst_index = np.argmax(simplex_values)
    return worst_index


def find_second_worst_index(simplex_values):
    sorted_indices = np.argsort(simplex_values)
    return sorted_indices[-2]


def find_centroid(simplex_points, worst_point_index):
    # reikia atmesti blogiausia simplexo taska
    valid_points = []
    n = len(simplex_points)
    for i in range(n):
        if i != worst_point_index:
            valid_points.append(simplex_points[i])

    # apskaiciuojame centroida
    centroid_x = sum(point[0] for point in valid_points) / len(valid_points)
    centroid_y = sum(point[1] for point in valid_points) / len(valid_points)
    return centroid_x, centroid_y


def reflection(centroid, worst_point, alpha=1):
    reflected_x = centroid[0] + alpha * (centroid[0] - worst_point[0])
    reflected_y = centroid[1] + alpha * (centroid[1] - worst_point[1])
    return reflected_x, reflected_y


def inner_contraction(centroid, worst_point, alpha=-0.5):  # perkelia taska i vidu (arciau centro)
    contracted_x = centroid[0] + alpha * (centroid[0] - worst_point[0])
    contracted_y = centroid[1] + alpha * (centroid[1] - worst_point[1])
    return contracted_x, contracted_y


def outer_contraction(centroid, worst_point, alpha=0.5):  # 
    contracted_x = centroid[0] + alpha * (centroid[0] - worst_point[0])
    contracted_y = centroid[1] + alpha * (centroid[1] - worst_point[1])
    return contracted_x, contracted_y


def shrink_simplex(f, simplex_points, gamma=0.5):
    # randam geriausia taska simplekse
    simplex_values = evaluate_simplex(f, simplex_points)
    best_point_index = find_best_index(simplex_values)
    best_point = simplex_points[best_point_index]

    # sukuriam nauja simplexa
    n = len(simplex_points)
    new_simplex = [best_point]

    for i in range(1, n):  # geriausias taskas jau idetas
        new_point = best_point + gamma * (simplex_points[i] - best_point)  # puse atstumo iki geriausio
        new_simplex.append(new_point)

    return new_simplex


def f_simplex(x1, x2):
    if x1 <= 0 or x2 <= 0:
        return 1 + abs(x1) + abs(x2)
    return -((1 - x1 - x2) * x1 * x2) / 8


def simplex(f, starting_point, tolerance):
    function_calls = 0
    steps = [starting_point]
    iteration = 0
    triangles = []

    # 1 sudaryti pradini simplexa
    simplex_points = generate_simplex_method_points(starting_point)
    function_calls += len(simplex_points)

    while True:
        # 2 visose simplexo virsunes apskaiciuoti tikslo funkcijos reiksmes
        simplex_values = evaluate_simplex(f, simplex_points)

        # 3 rasti didiausia f(x) reiksme ir ja atitinkanti virsune
        worst_point_index = find_worst_index(simplex_values)
        worst_point_value = simplex_values[worst_point_index]

        # randame geriausia simplexo taska
        best_point_index = find_best_index(simplex_values)
        best_point_value = simplex_values[best_point_index]

        # randam centroida
        centroid = find_centroid(simplex_points, worst_point_index)
        function_calls += 1

        # atliekame atspindi (refleksija)
        reflection_point = reflection(centroid, simplex_points[worst_point_index])
        function_calls += 1
        triangles.append(copy.deepcopy(simplex_points))
        reflection_value = f(reflection_point[0], reflection_point[1])

        # bandom daryti ekspansija
        if reflection_value <= best_point_value:  # refleksijos reiksme geresne nei simplexo best pointvalue
            expansion_point = reflection(centroid, reflection_point, 2)
            expansion_value = f(expansion_point[0], expansion_point[1])

            # ar ekspansijos taskas geresnis us simplex betpointvalue
            if expansion_value <= best_point_value:  # jei gautas naujas ekspansijos taskas yra geresnis, nei geriausiausio tasko
                simplex_points[worst_point_index] = expansion_point  # ekspansija pavyko
            else:
                simplex_points[worst_point_index] = reflection_point
                # ekspansijos taskas nera geresnis uz dabartini geriausia simplexo taska, refleksija buvo geresne nei geriausio tasko

        # tikrinam ar refleksijos taskas geresnis nei antras blogiausias dabartinis taskas
        elif reflection_value <= simplex_values[find_second_worst_index(simplex_values)]:
            simplex_points[worst_point_index] = reflection_point

        # kontrakcija(simplexo sutraukimas, perkelimas blogiausio tasko arciau geresniu)
        # vidine kontrakcija(kai reflektuotas taskas blogesnis uz simplexo blogiausia)
        elif reflection_value >= worst_point_value:
            inner_contraction_point = inner_contraction(centroid, simplex_points[worst_point_index])
            inner_contraction_value = f(inner_contraction_point[0], inner_contraction_point[1])

            function_calls += 1

            if inner_contraction_value <= worst_point_value:  # jei kontrakcijos taskas yra geresnis uz simplexo blogiausia
                simplex_points[worst_point_index] = inner_contraction_point
            else:  # kontrakcija nepavyko, atliekame simplexo sumazinima
                simplex_points = shrink_simplex(f, simplex_points)
                function_calls += 1

            # isorine kontrakcija
            # reflekcijos taskas yra geresnis uz blogiausia simplexo taska
        else:
            outer_contracion_point = outer_contraction(centroid, simplex_points[worst_point_index])
            outer_contracion_value = f(outer_contracion_point[0], outer_contracion_point[1])
            function_calls += 1

            if outer_contracion_value <= worst_point_value:
                simplex_points[worst_point_index] = outer_contracion_point

            else:  # shrink
                simplex_points = shrink_simplex(f, simplex_points)
                function_calls += 1

        simplex_values = evaluate_simplex(f, simplex_points)
        worst_point = np.array(simplex_points[find_worst_index(simplex_values)])
        best_point = np.array(simplex_points[find_best_index(simplex_values)])

        iteration += 1
        steps.append(best_point)

        if np.linalg.norm(worst_point - best_point) <= tolerance:
            break

    return simplex_points[find_best_index(simplex_values)], function_calls, iteration, steps, triangles, simplex_points


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
    ax.plot(x_points, y_points, 'bo-', label="Optimization Path")

    configurePlot(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename)
    if show:
        plt.show()
    plt.close()


def plot_triangles(triangles, filename, show=False, present=False):
    points = [[p[0], p[1], p[2], p[0]]
              for p in triangles]
    print(points)

    plt.figure("simplex")
    for point in points:
        xs = [x[0] for x in point]
        ys = [y[1] for y in point]
        plt.plot(xs, ys, '-o')

        if present:
            plt.savefig(filename)
            input("Press Enter to continue...")

    if not os.path.exists('figures'):
        os.mkdir('figures')
    plt.savefig("figures/" + filename)
    if show:
        plt.show()
    plt.close()


def plot_simplex_triangles(steps, triangles):
    plt.figure(figsize=(10, 8))

    for triangle in triangles:
        triangle_np = np.array(triangle)
        plt.fill(triangle_np[:, 0], triangle_np[:, 1], alpha=0.3)

    steps_np = np.array(steps)
    plt.plot(steps_np[:, 0], steps_np[:, 1], 'ro-', label='Geriausi taškai')
    plt.title("Simpleksas")
    plt.xlabel("X-ašis")
    plt.ylabel("Y-ašis")
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # plt.axis('equal')
    plt.show()


def main():
    # starting_point = (0, 0)
    # starting_point = (1 ,1)
    starting_point = (0.8, 0.8)

    epsilon = 0.001

    print(f"----------DEFORMUOJAMO SIMPLEKSO METODAS-----------")
    optimal_point, function_calls, iteration, steps, triangles, simplex_points = simplex(f, starting_point, epsilon)
    print(f"Optimalus taškas: {optimal_point}")
    print(f"Optimali funkcijos reikšmė: {f1(optimal_point)}")
    print(f"Funkcijų iškvietimo kiekis: {function_calls}")
    print(f"Iteracijos: {iteration}")
    print(math.dist(optimal_point, (1 / 3, 1 / 3)))
    # print([simplex_points])
    # plot(steps, "simplex", show=True)
    # plot_triangles(triangles + [simplex_points], "simplex_triangles", show = True)
    # plot_simplex_triangles(steps, triangles)
    # for i in range(0, len(steps)):
    #   print(f"{steps[i][0]}, {steps[i][1]},{f(steps[i][0],steps[i][1])}")


if __name__ == "__main__":
    main()
