import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


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
							label.append([0,1])
						else:
							label.append([1,0])
					data.append(l)
						
		return np.array(data), np.array(label)


def network_train(x_train, y_train, x_val, y_val):
	input_size = x_train.shape[1]
	hidden_size = 10
	output_size = 2

	x_train = Variable(torch.FloatTensor(x_train))
	y_train = Variable(torch.FloatTensor(y_train))
	x_val = Variable(torch.FloatTensor(x_val))
	y_val = Variable(torch.FloatTensor(y_val))

	model = nn.Sequential(nn.Linear(input_size, hidden_size),
						  nn.ReLU(),
						  nn.Linear(hidden_size, output_size),
						  nn.Softmax())

	criterion = torch.nn.BCELoss()

	optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)
	losses, accs = [], []
	for epoch in range(50):
		optimizer.zero_grad()
		y_pred = model(x_train)
		loss = criterion(y_pred, y_train)
		loss.backward()
		optimizer.step()

		print('epoch: ', epoch, ' loss: ', loss.item())
		losses.append(loss.item())
		if epoch % 10 == 0:
			acc_train = evaluate(y_pred, y_train)
			y_pred_val = model(x_val)
			acc_val = evaluate(y_pred_val, y_val)
			accs.append(acc_val)
			print('acc val: {}, train: {}'.format(acc_val.item(), acc_train.item()))
			if len(accs) != 0 and acc_val.item() > max(accs):
				torch.save(model, "./optimal_model.p")

	plot_data(losses, "training loss")
	plot_data(accs, "validation accuracy")


def evaluate(Y, Y_gt):
	Y = torch.argmax(Y, dim = 1)
	Y_gt = torch.argmax(Y_gt, dim = 1)
	result = (Y == Y_gt.long()).sum().float()
	return result/len(Y)


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


if __name__ == '__main__': 
	# Load Data
	path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
	path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
	path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')
	
	x_train, y_train = import_data(path_train)
	x_val, y_val = import_data(path_val)
	x_test, _ = import_data(path_test, test = True)
	network_train(x_train, y_train, x_val, y_val)
	load_model()