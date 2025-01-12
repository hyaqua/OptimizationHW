import numpy as np


class LinearModel:
    def __init__(self, a=np.empty([0, 0]), b=np.empty([0, 0]), c=np.empty([0, 0]), minmax="MIN"):
        self.a = a
        self.b = b
        self.c = c
        self.x = [float(0)] * len(c)
        self.minmax = minmax
        self.printIter = True
        self.optimalValue = None
        self.transform = False

    def addA(self, a):
        self.a = a

    def addB(self, b):
        self.b = b

    def addC(self, c):
        self.c = c
        self.transform = False

    def setObj(self, minmax):
        if minmax in ["MIN", "MAX"]:
            self.minmax = minmax
        else:
            print("Invalid objective.")
        self.transform = False

    def setPrintIter(self, print_iter):
        self.printIter = print_iter

    def printSoln(self):
        print("Optimal solution: ")
        solution = [f"{x:.2f}" for x in self.x]
        print(f"({', '.join(solution)})")

        print("Base:")
        base = [i + 1 for i, x in enumerate(self.x) if x != 0]
        print("{" + ", ".join(map(str, base)) + "}")

        print("Optimal value: ")
        print(f"{self.optimalValue:.2f}")

    def printTableau(self, tableau):
        print("\t\t", end="")
        for j in range(len(self.c)):
            print(f"x_{j} |", end="\t")
        for j in range(len(tableau[0]) - len(self.c) - 2):
            print(f"s_{j} |", end="\t")
        print()

        for row in tableau:
            for val in row[1:]:
                if not np.isnan(val):
                    print(f"{val:.2f} |", end="\t")
                else:
                    print("\t", end="")
            print()

    def getTableau(self):
        if self.minmax == "MIN" and not self.transform:
            self.c = -1 * self.c
            self.transform = True

        num_vars = len(self.c)
        num_slack = len(self.a)

        t1 = np.hstack(([None], [0], self.c, [0] * num_slack))

        basis = np.arange(num_vars, num_vars + num_slack)
        A = self.a

        if not (num_slack + num_vars == len(self.a[0])):
            A = np.hstack((self.a, np.identity(num_slack)))

        t2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))

        return np.array(np.vstack((t1, t2)), dtype='float')

    def simplexOptimization(self):
        if not self.transform:
            self.c = -1 * self.c

        tableau = self.getTableau()

        if self.printIter:
            print("Starting Tableau:")
            self.printTableau(tableau)

        for iteration in range(1, 51):
            if self.printIter:
                print("----------------------------------")
                print(f"Iteration: {iteration}")
                self.printTableau(tableau)

            if all(cost >= 0 for cost in tableau[0, 2:]):
                break

            pivot_col = np.argmin(tableau[0, 2:]) + 2

            ratios = []
            for i in range(1, len(tableau)):
                if tableau[i, pivot_col] > 0:
                    ratios.append((tableau[i, 1] / tableau[i, pivot_col], i))

            if not ratios:
                print("Problem is unbounded")
                return

            pivot_row = min(ratios)[1]
            pivot = tableau[pivot_row, pivot_col]

            print(f"Pivot Column: {pivot_col}")
            print(f"Pivot Row: {pivot_row}")
            print(f"Pivot Element: {pivot}")

            tableau[pivot_row, 1:] /= pivot
            for i in range(len(tableau)):
                if i != pivot_row:
                    mult = tableau[i, pivot_col] / tableau[pivot_row, pivot_col]
                    tableau[i, 1:] -= mult * tableau[pivot_row, 1:]

            tableau[pivot_row, 0] = pivot_col - 2

        if self.printIter:
            print("----------------------------------")
            print(f"Final Tableau reached in {iteration} iterations")
            self.printTableau(tableau)
        else:
            print("Solved")

        self.x = np.zeros(len(tableau[0, 2:]))
        for i in range(1, len(tableau)):
            if tableau[i, 0] < len(tableau[0, 2:]):
                self.x[int(tableau[i, 0])] = tableau[i, 1]

        self.optimalValue = -tableau[0, 1]


def main():
    model1 = LinearModel()

    A = np.array([[-1, 1, -1, -1],
                  [2, 4, 0, 0],
                  [0, 0, 1, 1]])
    b = np.array([1, 8, 8])
    c = np.array([2, -3, 0, -5])

    model1.addA(A)
    model1.addB(b)
    model1.addC(c)
    model1.setObj("MIN")
    model1.setPrintIter(True)

    print("A =\n", A, "\n")
    print("b =\n", b, "\n")
    print("c =\n", c, "\n\n")

    model1.simplexOptimization()
    print("\n")
    model1.printSoln()


if __name__ == "__main__":
    main()
