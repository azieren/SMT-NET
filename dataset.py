import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable


def import_data(path, test = False):
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
							label.append(1)
						else:
							label.append(-1)
					data.append(l)
						
		return np.array(data), np.array(label)


def network_train(x_train, y_train):
	input_size = x_train.shape[1]
	hidden_size = 1000
	output_size = 1

	x_train = Variable(torch.FloatTensor(x_train))
	y_train = Variable(torch.FloatTensor(y_train))

	model = nn.Sequential(nn.Linear(input_size, hidden_size),
						  nn.ReLU(),
						  nn.Linear(hidden_size, output_size),
						  nn.Sigmoid())

	criterion = torch.nn.MSELoss()

	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	for epoch in range(1000):
		y_pred = model(x_train)
		loss = criterion(y_pred, y_train)
		print('epoch: ', epoch, ' loss: ', loss.item())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


if __name__ == '__main__': 
	# Load Data
	path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
	path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
	path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')
	
	x_train, y_train = import_data(path_train)
	x_val, y_val = import_data(path_val)
	x_test, _ = import_data(path_test, test = True)
	network_train(x_train, y_train)