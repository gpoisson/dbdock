import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(13, 50) # 2 Input noses, 50 in middle layers
		self.fc2 = nn.Linear(50, 1) # 50 middle layer, 1 output nodes
		self.rl1 = nn.ReLU()
		self.rl2 = nn.ReLU()
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.rl1(x)
		x = self.fc2(x)
		x = self.rl2(x)
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

	batch_size = 4

	from sklearn import datasets
	digits = datasets.load_digits()

	ligands = np.load("features_all_norm.npy")
	labels = np.load("labels_all.npy")

	ligands = ligands[:,:13]

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

	trainingdataX = torch.Tensor(trainingdataX)
	trainingdataY = torch.Tensor(trainingdataY)

	losses = []

	NumEpoches = 1000
	import time
	for epoch in range(NumEpoches):

		running_loss = 0.0
		for i, data in enumerate(trainingdataX, 0):
			time.sleep(3.0)
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
			if i % 1000 == 0:
				print ("loss: ", running_loss)
				losses.append(running_loss)
				running_loss = 0.0
			print(inputs)
			print(optimizer)
			print(outputs)
			print(loss)
			print()
	print ("Finished training...")
	print (net(Variable(torch.FloatTensor(trainingdataX[0]))))

	plt.figure()
	plt.plot(range(len(losses)),losses,'x',ms=2,mew=3)
	plt.grid(True)
	plt.show()
	