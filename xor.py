import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

input_size = 34
hidden_layers = []
hidden_layers.append(200)	# first layer
hidden_layers.append(20)	# second layer
output_size = 1
batch_size = 4

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
		#self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
		self.fc2 = nn.Linear(hidden_layers[0], output_size)
		#self.fc3 = nn.Linear(hidden_layers[1], output_size) # 50 middle layer, 1 output nodes
		self.rl1 = nn.ReLU()
		#self.rl2 = nn.ReLU()
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.rl1(x)
		x = self.fc2(x)
		#x = self.rl2(x)
		#x = self.fc3(x)
		return x

if __name__ == "__main__":
	## Create Network

	net = Net()
	#print net

	## Optimization and Loss

	#criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
	criterion = nn.MSELoss()
	#criterion = nn.L1Loss()
	#criterion = nn.NLLLoss()
	#criterion = nn.BCELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
	#optimizer = optim.Adam(net.parameters(), lr=0.01)

	#trainingdataX = [[[0.01, 0.01], [0.01, 0.90], [0.90, 0.01], [0.95, 0.95]], [[0.02, 0.03], [0.04, 0.95], [0.97, 0.02], [0.96, 0.95]]]
	#trainingdataY = [[[0.01], [0.90], [0.90], [0.01]], [[0.04], [0.97], [0.98], [0.1]]]


	from sklearn import datasets
	digits = datasets.load_digits()

	ligands = np.load("features_all_norm_small.npy")
	labels = np.load("labels_all_small.npy")

	trainingdataX = [[]]
	trainingdataY = [[]]

	for sample in range(len(ligands)):
		trainingdataX[-1].append(ligands[sample])
		trainingdataY[-1].append([(float)(labels[sample])])
		if ((len(trainingdataX[-1])) >= batch_size):
			trainingdataX.append([])
			trainingdataY.append([])

	temp_x = []
	temp_y = []
	# trim out incomplete batches
	for batch in range(len(trainingdataX)):
		if (len(trainingdataX[batch]) == batch_size):
			temp_x.append(trainingdataX[batch])
			temp_y.append(trainingdataY[batch])
	
	trainingdataX = temp_x
	trainingdataY = temp_y
	'''
	val_dataX = trainingdataX[-120000:]
	trainingdataX = trainingdataX[:-120000]
	val_dataY = trainingdataY[-120000:]
	trainingdataY = trainingdataY[:-120000]

	trainingdataX = torch.Tensor(trainingdataX)
	trainingdataY = torch.Tensor(trainingdataY)
	val_dataX = torch.Tensor(val_dataX)
	val_dataY = np.asarray(val_dataY)
	'''
	#trainingdataX = trainingdataX.cuda(0)
	#trainingdataY = trainingdataY.cuda(0)

	losses = []

	NumEpoches = 20

	avg_running_loss = 0
	#import time
	for epoch in range(NumEpoches):
		running_loss = 0.0
		for i, data in enumerate(trainingdataX, 0):
			#time.sleep(3.0)
			inputs = data
			labels = trainingdataY[i]
			inputs = Variable(torch.FloatTensor(inputs))
			labels = Variable(torch.FloatTensor(labels))
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()        
			optimizer.step()
			running_loss += loss.data[0]
			if ((i > 10) & (i % NumEpoches == 0)):
				avg_running_loss = running_loss / NumEpoches
				losses.append(avg_running_loss)
				print ("prediction: {}   loss: {}    avg loss per sample: {}".format(outputs.data, running_loss, avg_running_loss))
				running_loss = 0.0
			'''
			print(inputs)
			print(optimizer)
			print(outputs)
			print(loss)
			print()
			'''
	print ("Finished training...")
	test_sample = np.asarray(trainingdataX[0])
	test_label = np.asarray(trainingdataY[0])
	print (net(Variable(torch.FloatTensor(test_sample))))
	print(test_label)
	'''
	predictions = []
	for batch in range(len(val_dataX)):
		predictions.append([net(Variable(torch.FloatTensor(val_dataX[batch])))])
	predictions = np.asarray(predictions)
	print(len(predictions))
	print(len(predictions[0]))
	print(len(predictions[0][0]))
	print(len(predictions[0][0][0]))
	print("predictions[0]: {}".format(predictions[0]))
	print("val_dataY[0]: {}".format(val_dataY[0]))
	errors = abs(predictions - val_dataY)
	print("errors[0]: {}".format(errors[0]))
	sq_errors = errors ** 2
	sum_sq_errors = np.sum(sq_errors)
	res_errors = abs(val_dataY - np.mean(val_dataY)) ** 2
	sum_res_sq = np.sum(res_errors)
	r2 = 1 / (sum_sq_errors - sum_res_sq)
	print("r2: {}".format(r2))
	'''
	plt.figure()
	plt.plot(range(len(losses)),losses,'x',ms=2,mew=3)
	plt.grid(True)
	plt.suptitle("Neural Network {}-{}-{}".format(input_size,hidden_layers[0],output_size))
	plt.xlabel("Training Samples")
	plt.ylabel("Training Error (kcal/mol)")
	plt.show()
	