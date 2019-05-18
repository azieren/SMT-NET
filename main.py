import os
from pysmt.shortcuts import Symbol, And, GE, LT, Plus, Equals, Int, Real, Solver
from pysmt.typing import INT, REAL


if __name__ == '__main__':
    hello = [Symbol(s, REAL) for s in "hello"]
    world = [Symbol(s, REAL) for s in "world"]

    letters = set(hello+world)

    # All letters between 1 and 10
    domains = And([And(GE(l, Real(1)),
                    LT(l, Real(10))) for l in letters])

    sum_hello = Plus(hello) # n-ary operators can take lists
    sum_world = Plus(world) # as arguments
    problem = And(Equals(sum_hello, sum_world), Equals(sum_hello, Real(10)))
    formula = And(domains, problem)

    print("Serialization of the formula:")
    print(formula)


    with Solver(name="msat") as solver:
        solver.add_assertion(domains)
        if not solver.solve():
            print("Domain is not SAT!!!")
            exit()
        solver.add_assertion(problem)
        if solver.solve():
            for l in letters:
                print("%s = %s" %(l, solver.get_value(l)))
        else:
            print("No solution found")
