import os
from pysmt.shortcuts import Symbol, And, GE, LT, Plus, Equals, Int, Real, Solver, Or
from pysmt.typing import INT, REAL, ArrayType


if __name__ == '__main__':
    weight_1 = [[0.] * 10] * 784
    bias_1 = [0.] * 10
    weight_2 = [[0.]*2]*10
    bias_2 = [0.] * 2

    weight_1 = [Symbol("w1 " + str(i) + " " + str(j), REAL) for i in range(len(weight_1)) for j in range(len(weight_1[i]))]
    bias_1 = [Symbol("b1 " + str(i), REAL) for i in range(len(bias_1))]
    weight_2 = [Symbol("w2 " + str(i) + " " + str(j), REAL) for i in range(len(weight_2)) for j in range(len(weight_2[i]))]
    bias_2 = [Symbol("b2 " + str(i), REAL) for i in range(len(bias_2))]

    letters = set(weight_1+bias_1+weight_2+bias_2)

    # All letters between 1 and 10
    domains = And([And(GE(l, Real(-1)), LT(l, Real(1))) for l in letters])
    formula = And(domains)

    print("Serialization of the formula:")
    print(formula)


    with Solver(name="msat") as solver:
        solver.add_assertion(domains)
        if not solver.solve():
            print("Domain is not SAT!!!")
            exit()
        if solver.solve():
            for l in letters:
                print("%s = %s" %(l, solver.get_value(l)))
        else:
            print("No solution found")
