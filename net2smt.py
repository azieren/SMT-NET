import os
from pysmt.shortcuts import Symbol, And, GE, LT, Plus, Equals, Int, Real, Solver, Or, Times, Pow
from pysmt.typing import INT, REAL, ArrayType
from dataset import import_data


class SMTNet:
    def __init__(self, n, h):
        self.n = n
        self.dim_list = [n] + h + [2]
        self.create_weights()
    
    def create_weights(self):
        self.net = {}
        for i, h in enumerate(self.dim_list[:-1]):
            weight = [[Symbol("w{}_{}_{}".format(i, j, k), REAL) for j in range(h)] for k in range(self.dim_list[i+1])]
            bias = [Symbol("b{}_{}" .format(i, j), REAL) for j in range(self.dim_list[i+1])]
            self.net[i] = (weight, bias)
        return
    
    def regularize(self, l = 0.5):
        
        w_reg_list = []
        for i, (weight, _) in self.net.items(): 
            print(i)
            w_reg_list.append(Plus([Pow(w, Int(2)) for w_r in weight for w in w_r]))           
            print(w_reg_list[-1])
        regularize = And([And(GE(w, Real(-l)), LT(w, Real(l))) for w in w_reg_list])
        return regularize
    
    def feed_data(self, X, Y):
        formula = []
        for x, y in zip(X, Y):
            x = x[100:103]
            x_formula = []
            for i, (weight, bias) in self.net.items(): 
                if i == 0:
                    x_hidden = []
                    for r, w_r in enumerate(weight):
                        x_hidden.append(Plus([Plus(Times(w, Real(float(x[c]))), bias[r]) for c, w in enumerate(w_r)]))
                    x = x_hidden
                else:
                    x_hidden = []
                    for r, w_r in enumerate(weight):
                        x_hidden.append(Plus([Plus(Times(w, x[c]), bias[r]) for c, w in enumerate(w_r)]))
                    x = x_hidden
                ## Add activation function
            if np.argmax(y) == 0:
                x_formula.append(And(GE(x[0], Real(0.5)), LT(x[1], Real(0.5))))
            else:
                x_formula.append(And(GE(x[1], Real(0.5)), LT(x[0], Real(0.5))))
                
            x_formula.append(x)
        return And(x_formula)

   

if __name__ == '__main__':
    path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
    path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
    path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')

    x_train, y_train = import_data(path_train)
    #x_val, y_val = import_data(path_val)
    #x_test, y_test = import_data(path_test, test = True)

    N, n = x_train.shape
    
    """weight_1 = [Symbol("w1_" + str(i) + "_" + str(j), REAL) for i in range(n) for j in range(h1)]
    bias_1 = [Symbol("b1_" + str(i), REAL) for i in range(h1)]
    weight_2 = [Symbol("w2_" + str(i) + "_" + str(j), REAL) for i in range(h1) for j in range(2)]
    bias_2 = [Symbol("b2_" + str(i), REAL) for i in range(2)]


    print(weight_1)"""

    smt_net = SMTNet(3, [4])

    regularize = smt_net.regularize()
    print(regularize)
    
    formula = And(regularize)

    print("Serialization of the formula:")
    #print(formula)


    with Solver(name="msat") as solver:
        solver.add_assertion(formula)
        if not solver.solve():
            print("Domain is not SAT!!!")
            exit()
        if solver.solve():
            for l in letters:
                print("%s = %s" %(l, solver.get_value(l)))
        else:
            print("No solution found")
