import numpy as np


def getpoint(arg, value):
    return {"arg": arg, "value": value}


def getmodvector(x):
    return (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5


def simplex_method(func, args, epsilon=0.001, alpha=0.5, gama=3, beta=0.5, niu=0.5, max_iterations=1000):
    simplex = [getpoint(args, func(args))]
    counter = 1

    for i in range(0, len(args)):
        arg_list = list(args)
        arg_list[i] += alpha
        simplex.append(getpoint(arg_list, func(arg_list)))
        counter += 1

    for i in range(0, max_iterations):
        # 1. Sort
        simplex.sort(key=lambda x: x['value'])

        # 6. Check convergence
        if getmodvector(np.array((simplex[0]['arg']) - np.array(simplex[-1]['arg']))) < epsilon:
            break

        centroid = [0] * len(args)
        for j in range(0, len(args)):
            for k in range(0, len(simplex) - 1):
                centroid[j] += simplex[k]['arg'][j]
            centroid[j] /= (len(simplex) - 1)

        # 2. Reflect
        reflection = [0] * len(args)
        for j in range(0, len(args)):
            reflection[j] = centroid[j] + alpha * (centroid[j] - simplex[-1]['arg'][j])
        reflection_value = func(reflection)
        counter += 1

        # 3. Evaluate or Extend
        if simplex[0]['value'] <= reflection_value < simplex[-2]['value']:
            simplex[-1] = getpoint(reflection, reflection_value)
            continue
        elif reflection_value < simplex[0]['value']:
            extend = [0] * len(args)
            for j in range(0, len(args)):
                extend[j] = centroid[j] + gama * (reflection[j] - centroid[j])
            extended_value = func(extend)
            counter += 1
            if extended_value < simplex[0]['value']:
                simplex[-1] = getpoint(extend, extended_value)
            else:
                simplex[-1] = getpoint(reflection, reflection_value)
            continue

        # 4. Contract
        contraction = [0] * len(args)
        for j in range(0, len(args)):
            contraction[j] = centroid[j] + niu * (simplex[-1]['arg'][j] - centroid[j])
        contraction_value = func(contraction)
        counter += 1
        if contraction_value < simplex[-1]['value']:
            simplex[-1] = getpoint(contraction, contraction_value)
            continue

        # 5. Reduce
        for j in range(1, len(simplex)):
            reduce = [0] * len(args)
            for k in range(0, len(args)):
                reduce[k] = simplex[0]['arg'][k] + beta * (simplex[j]['arg'][k] - simplex[0]['arg'][k])
            reduce_value = func(reduce)
            counter += 1
            simplex[j] = getpoint(reduce, reduce_value)
    return simplex[0]['arg'], counter


def optimization(func, constraints, args, epsilon, penalty_quantifier=1, rdiv=0.5, pretty_print=False, to_csv=False):
    penalty_print = penalty_quantifier
    penalty_quantifier = penalty_quantifier / rdiv

    def penalty_function(x):
        return sum(eq(x) ** 2 for eq in constraints[0]) + sum(max(0, iq(x)) ** 2 for iq in constraints[1])

    def r(x):
        return x * rdiv

    def bfunc(x):
        return func(x) + 1 / r(penalty_quantifier) * penalty_function(x)

    counter = 0
    max_iterations = 100
    newargs = None
    i = None
    if to_csv:
        print("Iteracija;X;B(X,r);f(X);r")
    for i in range(1, max_iterations):
        newargs, c = simplex_method(bfunc, args, epsilon)
        counter += c

        if pretty_print:
            print(i)
            print("r: ", r(penalty_quantifier))
            print("Baudos funkcija:", bfunc(newargs))
            print("X:", "(" + str(newargs[0]) + ", " + str(newargs[1]) + ", " + str(newargs[2]) + ")")
            print("F(X): ", func(newargs))
            print('*************')
        elif to_csv:
            print(str(i) + "; (" + str(newargs[0]) + ", " + str(newargs[1]) + ", " + str(newargs[2]) + ");"
                  + str(bfunc(newargs)) + ";" + str(func(newargs)) + ";" + str(r(penalty_quantifier)))
        if getmodvector(np.array(newargs) - np.array(args)) < epsilon:
            break
        else:
            args = newargs
            penalty_quantifier = r(penalty_quantifier)

    # if not to_csv:
    #     print("X:", newargs, ". f(X):", func(newargs), ". iterations:", i)
    #     print("counter: ", counter)
    # print(f"Gauname, kad prireikė {i} iteracijų ir {counter} funkcijos iškvietimų.")
    if not to_csv:
        print(f"{penalty_print};{i};({newargs[0]}, {newargs[1]}, {newargs[2]});{func(newargs)};{counter}")

# funkcija yra Turis, gi yra (pavirsiaus plotas - 1)
def funkcija(x):
    return -1 * x[0] * x[1] * x[2]


def g1(x):
    return 2 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2]) - 1


def h1(x):
    return -x[0]


def h2(x):
    return -x[1]


def h3(x):
    return -x[2]


def main():
    equality = [g1]
    inequality = [h1, h2, h3]
    constraints = [equality, inequality]
    x_0 = [0, 0, 0]
    x_1 = [1, 1, 1]
    x_m = [0.1, 0.8, 0.8]
    # print("Funkcija taske 0,0,0:", funkcija(x_0))
    # print("Funkcija taske 1,1,1:", funkcija(x_1))
    # print("Funkcija taske 0.1,0.8,0.8:", funkcija(x_m))
    #
    # print("g(X) taske 0,0,0:", g1(x_0))
    # print("g(X) taske 1, 1, 1:", g1(x_1))
    # print("g(X) taske 0.1,0.8,0.8:", g1(x_m))
    #
    # print("h1(X) taske 0,0,0:", h1(x_0))
    # print("h2(X) taske 0,0,0:", h2(x_0))
    # print("h3(X) taske 0,0,0:", h3(x_0))
    #
    # print("h1(X) taske 1,1,1:", h1(x_1))
    # print("h2(X) taske 1,1,1:", h2(x_1))
    # print("h3(X) taske 1,1,1:", h3(x_1))
    #
    # print("h1(X) taske 0.1,0.8,0.8:", h1(x_m))
    # print("h2(X) taske 0.1,0.8,0.8:", h2(x_m))
    # print("h3(X) taske 0.1,0.8,0.8:", h3(x_m))

    # optimization(funkcija, constraints, x_0, 0.0001, penalty_quantifier=4, rdiv=0.5)
    # optimization(funkcija, constraints, x_1, 0.0001, penalty_quantifier=4, rdiv=0.5)
    q=[9,17,1,0.5,0.2]
    for i in q:
        optimization(funkcija, constraints, x_m, 0.0001, penalty_quantifier=i, rdiv=0.15)


if __name__ == "__main__":
    main()
