import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys

input_size = 34						# size of input features


if (len(sys.argv) == 2):
	hidden_layers = [(int)(sys.argv[1])]
elif (len(sys.argv) == 3):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2]))]				# list of hidden layer dimensions
elif (len(sys.argv) == 4):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2])),((int)(sys.argv[3]))]				# list of hidden layer dimensions

'''
hidden_layers = []
for layer in range((int)(sys.argv[1])):
	hidden_layers.append((int)(sys.argv[layer+1]))
'''

output_size = 1						# size of output features
batch_size = 5						# number of samples per batch

training_set_size = 5000				# number of samples in training set
test_set_size = 4000				# number of samples in test set

NumEpoches = 5


def get_data():
	ligands = np.load("features_all_norm.npy")
	labels = np.load("labels_all.npy")

	#ligands = ligands[:,:13]

	if (test_set_size + training_set_size > len(ligands)):
		print("TEST SET SIZE + TRAINING SET SIZE: {} SAMPLES\nTOTAL SAMPLES AVAILABLE: {}".format((test_set_size + training_set_size),len(ligands)))
	
	train_ligands = ligands[:training_set_size]
	train_labels = labels[:training_set_size]
	test_ligands = ligands[-test_set_size:]
	test_labels = labels[-test_set_size]

	train_ligands = np.random.shuffle(train_ligands)
	train_labels = np.random.shuffle(train_labels)
	test_ligands = np.random.shuffle(test_ligands)
	test_labels = np.random.shuffle(test_labels)

	trainingdataX = [[]]
	trainingdataY = [[]]

	for sample in range(len(train_ligands)):
		trainingdataX[-1].append(train_ligands[sample])
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

	testdataX = [[]]
	testdataY = [[]]

	for sample in range(len(test_ligands)):
		testdataX[-1].append(test_ligands[sample])
		testdataY[-1].append([(float)(labels[sample])])
		if ((len(testdataX[-1])) >= batch_size):
			testdataX.append([])
			testdataY.append([])

	temp_x = []
	temp_y = []
	# trim out incomplete batches
	for batch in range(len(testdataX)):
		if (len(testdataX[batch]) == batch_size):
			temp_x.append(testdataX[batch])
			temp_y.append(testdataY[batch])
	
	testdataX = temp_x
	testdataY = temp_y

	return trainingdataX, trainingdataY, testdataX, testdataY

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		if (len(hidden_layers) == 1):
			self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
			self.fc2 = nn.Linear(hidden_layers[0], output_size)
			self.rl1 = nn.ReLU()
		elif (len(hidden_layers) == 2):
			self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
			self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
			self.fc3 = nn.Linear(hidden_layers[1], output_size)
			self.rl1 = nn.ReLU()
			self.rl2 = nn.ReLU()			
		elif (len(hidden_layers) == 3):
			self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
			self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
			self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
			self.fc4 = nn.Linear(hidden_layers[2], output_size) # 50 middle layer, 1 output nodes
			self.rl1 = nn.ReLU()
			self.rl2 = nn.ReLU()
			self.rl3 = nn.ReLU()
		
	def forward(self, x):
		if (len(hidden_layers) == 1):
			x = self.fc1(x)
			x = self.rl1(x)
			x = self.fc2(x)
		elif (len(hidden_layers) == 2):
			x = self.fc1(x)
			x = self.rl1(x)
			x = self.fc2(x)
			x = self.rl2(x)
			x = self.fc3(x)
		elif (len(hidden_layers) == 3):
			x = self.fc1(x)
			x = self.rl1(x)
			x = self.fc2(x)
			x = self.rl2(x)
			x = self.fc3(x)
			x = self.rl3(x)
			x = self.fc4(x)

		return x

def main():
	## Create Network

	net = Net()

	## Optimization and Loss

	#criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
	criterion = nn.MSELoss()
	#criterion = nn.L1Loss()
	#criterion = nn.NLLLoss()
	#criterion = nn.BCELoss()
	#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
	optimizer = optim.Adam(net.parameters(), lr=0.01)

	trainingdataX, trainingdataY, testdataX, testdataY = get_data()

	'''
	trainingdataX = trainingdataX.cuda()
	trainingdataY = trainingdataY.cuda()
	testdataX = testdataX.cuda()
	testdataY = testdataY.cuda()
	'''

	losses = []

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
			if ((i > 10) & (i % 20 == 0)):
				avg_running_loss = running_loss / NumEpoches
				losses.append(avg_running_loss)
				#print ("prediction: {}   actual: {}   loss: {}    avg loss per sample: {}".format(outputs.data.numpy(), labels, running_loss, avg_running_loss))
				running_loss = 0.0
			'''
			print(inputs)
			print(optimizer)
			print(outputs)
			print(loss)
			print()
			'''
	#print ("Finished training...")

	predictions = []
	actual = []
	losses = []
	for sample in range(len(testdataX)):
		batch_prediction = net(Variable(torch.FloatTensor(testdataX[sample])))
		batch_prediction = batch_prediction.data.numpy()
		for p in range(len(batch_prediction)):
			predictions.append(batch_prediction[p])
			actual.append(testdataY[sample][p])
			losses.append(abs(predictions[-1] - testdataY[sample][p]))

	from sklearn.metrics import r2_score
	r2 = r2_score(actual,predictions)
	#print("r2: {}".format(r2))

	#print("Hidden Layers: {}   Median Error on Test Data: {} St Dev: {}   r2: {}".format(hidden_layers,np.mean(losses),np.std(losses),r2))
	#print("hidden_layers {} r2 {}   pred: {}   actual: {}".format(hidden_layers,r2,predictions[10],actual[10]))
	h = "{}-".format(input_size)
	for hid in hidden_layers:
		h += "{}-".format(hid)
	h += "{}".format(output_size)
	ex = 150
	print("hid_layers: {} r2: {} pred[ex]: {} actl[ex]: {} train_size: {} test_size: {} epochs: {} batch_size: {}".format(h,r2,predictions[ex],actual[ex],training_set_size,test_set_size,NumEpoches,batch_size))

	plt.figure()
	plt.plot(actual,predictions,'x',ms=2,mew=3)
	plt.grid(True)
	h_lays = ""
	for i in hidden_layers:
		h_lays += "{}-".format(i)
	plt.suptitle("Neural Network {}-{}{}  Median Error: {}\nBatch Size: {}  Epochs: {}  Train Set: {}   r2: {}".format(input_size,h_lays,output_size,np.median(losses),batch_size,NumEpoches,training_set_size,r2))
	plt.ylim([0,-10])
	plt.xlim([0,-10])
	plt.xlabel("Actual Values")
	plt.ylabel("Predicted Values (kcal/mol)")
	plt.show()
	'''
	plt.figure()
	plt.plot(range(len(losses)),losses,'x',ms=2,mew=3)
	plt.grid(True)
	h_lays = ""
	for i in hidden_layers:
		h_lays += "{}-".format(i)
	plt.suptitle("Neural Network {}-{}{}  Median Error: {}\nBatch Size: {}  Epochs: {}  Train Set: {}   St. Dev.: {}".format(input_size,h_lays,output_size,np.median(losses),batch_size,NumEpoches,training_set_size,np.std(losses)))
	plt.ylim([0,10])
	plt.xlabel("Test Batches")
	plt.ylabel("Mean Test Error (kcal/mol)")
	plt.show()
	'''
	
main()