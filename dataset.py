import os
import numpy as np
import matplotlib.pyplot as plt

def import_data (path, test = False):
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
        
if __name__ == '__main__': 
	# Load Data
	path_train = os.path.join(os.path.dirname(__file__), 'pa2_train.csv')
	path_val = os.path.join(os.path.dirname(__file__), 'pa2_valid.csv')
	path_test = os.path.join(os.path.dirname(__file__), 'pa2_test_no_label.csv')
	
	x_train, y_train = import_data(path_train)
	x_val, y_val = import_data(path_val)
	x_test, _ = import_data(path_test, test = True)
	

	#x_train = x_train[:1000]
	#y_train = y_train[:1000]
	print(len(x_train[0]), x_train[0][-1], len(y_train))
	dim = len(x_train)