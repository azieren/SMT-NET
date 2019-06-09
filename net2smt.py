import os
import numpy as np
from pysmt.shortcuts import Symbol, And, GE, LT, Plus, Equals, Int, Real, Solver, Or, Times, Pow, NotEquals, Max
from pysmt.shortcuts import is_sat, get_formula_size
from pysmt.typing import INT, REAL, ArrayType
from network import import_data
import network

class SMTNet:
    def __init__(self, n, h, activation='relu'):
        self.activation = activation
        self.n = n
        self.dim_list = [n] + h + [2]
        self.net_formula = {}
        self.net = {}
        self.create_weights()
    
    def create_weights(self):
        for i, h in enumerate(self.dim_list[:-1]):
            weight = [[Symbol("w{}_{}_{}".format(i, j, k), REAL) for j in range(h)] for k in range(self.dim_list[i+1])]
            bias = [Symbol("b{}_{}" .format(i, j), REAL) for j in range(self.dim_list[i+1])]
            self.net_formula[i] = (weight, bias)
            self.net[i] = None
        return
    
    def regularize(self, l = 0.5):
        w_reg_list = []
        for i, (weight, _) in self.net_formula.items():
            #print(i)
            w_reg_list.append(Plus([Pow(w, Real(2)) for w_r in weight for w in w_r]))
            #print(w_reg_list[-1])
        regularize = And([And(GE(w, Real(-l)), LT(w, Real(l))) for w in w_reg_list])
        return regularize

    def non_zero(self):
        w_reg_list = []
        for (weights,bias) in self.net_formula.values():
            for w_r,b in zip(weights, bias):
                for w in w_r:
                    coin = np.random.uniform(0,1,1)
                    if coin > 0.5:
                        w_reg_list.append(NotEquals(w, Real(0)))
                #w_reg_list.append(NotEquals(b, Real(0)))
        regularize = And(w_reg_list)
        print(regularize)
        return regularize

    def feed_data(self, X, Y):
        formula = []
        for x, y in zip(X, Y):
            x_formula = []
            for i, (weight, bias) in self.net_formula.items():
                if i == 0:
                    x_hidden = []
                    for r, w_r in enumerate(weight):
                        tmp = Plus([Plus(Times(w, Real(float(x[c]))), bias[r]) for c, w in enumerate(w_r)])
                        if self.activation == 'relu':
                            x_hidden.append(Max(tmp, Real(0)))
                        else:
                            x_hidden.append(tmp)
                        """node_output = []
                        for c, w in enumerate(w_r):
                            var = Plus(Times(w, Real(float(x[c]))), bias[r])
                            exp_3 = Pow(var, Real(3))
                            #exp_5 = Pow(var, Real(5))
                            sen_2 = Times(Real(0.25), var)
                            sen_3 = Times(Real(0.02), exp_3)
                            #sen_4 = Times(Real(0.002), exp_5)
                            node_output.append(Plus(Real(0.5), sen_2, sen_3))#, sen_4))
                        matrix_addition = Plus(node_output[i] for i in range(len(node_output)))
                        x_hidden.append(matrix_addition)"""
                    x = x_hidden
                else:
                    x_hidden = []
                    for r, w_r in enumerate(weight):
                        tmp = Plus([Plus(Times(w, x[c]), bias[r]) for c, w in enumerate(w_r)])
                        if self.activation == 'relu' and i < len(self.net_formula)-1:
                            x_hidden.append(Max(tmp, Real(0)))
                        else:
                            x_hidden.append(tmp)
                        """node_output = []
                        for c, w in enumerate(w_r):
                            var = Plus(Times(w, x[c]), bias[r])
                            exp_3 = Pow(var, Real(3))
                            #exp_5 = Pow(var, Real(5))
                            sen_2 = Times(Real(0.25), var)
                            sen_3 = Times(Real(0.02), exp_3)
                            #sen_4 = Times(Real(0.002), exp_5)
                            node_output.append(Plus(Real(0.5), sen_2, sen_3)) #, sen_4))
                        matrix_addition = Plus(node_output[i] for i in range(len(node_output)))
                        x_hidden.append(matrix_addition)"""
                    x = x_hidden
                ## Add activation function
            if np.argmax(y) == 0:
                x_formula.append(GE(x[0], x[1]))
            else:
                x_formula.append(GE(x[1], x[0]))
        return And(x_formula)

    def solve(self, formula):
        print("Serialization of the formula:")
        with Solver(name="z3") as solver:
            solver.add_assertion(formula)
            if not solver.solve():
                print("Domain is not SAT!!!")
                exit()
            if solver.solve():
                for i, (weight, bias) in self.net_formula.items():
                    w_curr = []
                    b_curr = []
                    for w_row, b in zip(weight, bias):
                        w_r = []
                        for w in w_row:
                            value = float(solver.get_value(w).constant_value())
                            w_r.append(value)
                        b_curr.append(float(solver.get_value(b).constant_value()))
                        w_curr.append(w_r)
                    self.net[i] = (np.array(w_curr, dtype=float).T, np.array(b_curr, dtype=float))
            else:
                print("No solution found")

    def test(self, X, Y):
        tp, fp, tn, fn = 0, 0, 0, 0
        for x, y in zip(X, Y):
            output = x
            for i,(w,b) in self.net.items():
                output = np.dot(w.T, output) + b
            y_gt, y_pred = np.argmax(y), np.argmax(output)
            if y_gt == 0 and y_pred == 0:
                tp += 1
            elif y_gt == 1 and y_pred == 0:
                fn += 1
            elif y_gt == 0 and y_pred == 1:
                fp += 1
            else:
                tn += 1
        acc = 1.0*(tn + tp)/(tp + tn + fp + fn)
        return acc

if __name__ == '__main__':
    acc_train, acc_val = 0.0, 0.0
    path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
    path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
    path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')

    x_train, y_train = import_data(path_train)
    x_val, y_val = import_data(path_val)
    x_test, _ = import_data(path_test, test = True)

    #x_train = np.array([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
    #y_train = np.array([[1,0], [0,1], [0,1], [1,0]])

    N, n = x_train.shape
    smt_net = SMTNet(n, [10])

    formula = smt_net.feed_data(x_train[:100], y_train[:100])
    regularize = smt_net.non_zero()
    #regularize = smt_net.regularize()
    formula = And(formula, regularize)
    size = get_formula_size(formula)
    print(size)
    
    smt_net.solve(formula)

    weights = []
    for i,(w,b) in smt_net.net.items():
        print("Weight: ", i, w)
        print("Bias: ", i, b)
        weights.append((i, w))

    network.main_network(weights)

    #acc_train = smt_net.test(x_train, y_train)
    acc_train = smt_net.test(x_train[:200], y_train[:200])
    #acc_val = smt_net.test(x_val, y_val)

    print("Train accuracy: {} \nValidation accuracy: {}:".format(acc_train, acc_val))



