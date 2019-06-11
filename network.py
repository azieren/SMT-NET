import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import pickle

def import_data(path, test=False):
    data = []
    label = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.split(',')
            l = [float(x) for x in l]
            if not test:
                gt = int(l.pop(0))
                if gt == 3:
                    label.append([0, 1])
                else:
                    label.append([1, 0])
            data.append(l)

    return np.array(data), np.array(label)


def init_weights(m):
     print("In init_weights, submodule:", m)
     if type(m) == nn.Linear:
         m.weight.data.fill_(1.0)
         print(m.weight)


def network_train(x_train, y_train, x_val, y_val, initialized_weights=None, initialized_biases=None):
    input_size = x_train.shape[1]
    hidden_size = 50
    output_size = 2

    x_train = Variable(torch.FloatTensor(x_train))
    y_train = Variable(torch.FloatTensor(y_train))
    x_val = Variable(torch.FloatTensor(x_val))
    y_val = Variable(torch.FloatTensor(y_val))

    model = nn.Sequential(nn.Linear(input_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, output_size),
                          nn.Softmax())

    with torch.no_grad():
        if initialized_weights is not None:
            for item_i, item in enumerate(initialized_biases):
                for i in range(len(item.T)):
                    model[item_i * 2].bias.data[i].fill_(float(item.T[i]))
            for item_i, item in enumerate(initialized_weights):
                for i in range(len(item.T)):
                    for j in range(len(item.T[i])):
                        model[item_i * 2].weight.data[i][j].fill_(float(item.T[i][j] * 0.01))

    for name, param in model.named_modules():
        param.requires_grad = True

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    losses, val_accs, train_accs = [], [], []
    for epoch in range(50):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()

        for param in model.parameters():
            print("gradients: ", param.grad.data.sum())

        optimizer.step()

        print('epoch: ', epoch, ' loss: ', loss.item())
        losses.append(loss.item())

        # if epoch % 5 == 0:

        acc_train = evaluate(y_pred, y_train)
        y_pred_val = model(x_val)
        acc_val = evaluate(y_pred_val, y_val)
        val_accs.append(acc_val)
        train_accs.append(acc_train)
        print('acc val: {}, train: {}'.format(acc_val.item(), acc_train.item()))
        if len(val_accs) != 0 and acc_val.item() >= max(val_accs):
            torch.save(model, "./optimal_model.p")
            print("Best model saved!")

    plot_data(losses, "training loss")
    plot_data(val_accs, "validation accuracy")
    plot_data(train_accs, "training accuracy")


def evaluate(Y, Y_gt):
    Y = torch.argmax(Y, dim = 1)
    Y_gt = torch.argmax(Y_gt, dim = 1)
    result = (Y == Y_gt.long()).sum().float()
    return result / len(Y)
    # result = (Y_gt == Y).data.sum() / 2
    # return float(result) / float(len(Y))


def load_model():
    model = torch.load("./optimal_model.p")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            print(name, param.data.shape)


def plot_data(value, title):
    plt.title(title)
    plt.plot(value)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.savefig(title + '.png')
    plt.clf()


def main_network(initialized_weights, initialized_biases):
    # pickle.dump(initialized_weights, open("./100_10_10/100_weights.p", "wb"))
    # pickle.dump(initialized_biases, open("./100_10_10/100_biases.p", "wb"))
    # initialized_weights, initialized_biases = 0, 0
    initialized_weights = pickle.load(open(r"./100_50/100_weights.p", "rb"))
    initialized_biases = pickle.load(open(r"./100_50/100_biases.p", "rb"))


    # Load Data
    path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
    path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
    path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')

    x_train, y_train = import_data(path_train)

    # x_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    # y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    x_val, y_val = import_data(path_val)
    x_test, _ = import_data(path_test, test=True)
    # network_train(x_train, y_train, x_val, y_val, None)
    network_train(x_train, y_train, x_val, y_val, initialized_weights, initialized_biases)
    load_model()


main_network(None, None)
