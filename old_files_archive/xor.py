import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVR

input_size = 20						# size of input features
if (len(sys.argv) == 2):
	hidden_layers = [(int)(sys.argv[1])]
elif (len(sys.argv) == 3):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2]))]				# list of hidden layer dimensions
elif (len(sys.argv) == 4):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2])),((int)(sys.argv[3]))]				# list of hidden layer dimensions

#hidden_layers = [100]

output_size = 1						# size of output features
batch_size = 5						# number of samples per batch
training_set_size = 2000				# number of samples in training set
test_set_size = 1000				# number of samples in test set
NumEpoches = 60
learning_rate = 0.012

C = 100000.0
epsilon = 0.01


def get_data():
	ligands = np.load("features_all_norm.npy")
	labels = np.load("labels_all.npy")

	ligands = ligands[:,14:34]

	if (test_set_size + training_set_size > len(ligands)):
		print("TEST SET SIZE + TRAINING SET SIZE: {} SAMPLES\nTOTAL SAMPLES AVAILABLE: {}".format((test_set_size + training_set_size),len(ligands)))
	
	train_ligands = ligands[:training_set_size]
	train_labels = labels[:training_set_size]
	test_ligands = ligands[-test_set_size:]
	test_labels = labels[-test_set_size:]

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
		testdataY[-1].append([(float)(test_labels[sample])])
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

def permute_data(X,y):
	samples = []
	labels = []

	# unpack the batches
	for batch in range(len(X)):
		for s in range(len(X[batch])):
			samples.append(X[batch][s])
			labels.append(y[batch][s])

	# define a fixed permutation
	permutation = np.random.permutation(len(samples))

	# shuffle the data
	shuff_samples = []
	shuff_labels = []

	for s in range(len(samples)):
		shuff_samples.append(samples[permutation[s]])
		shuff_labels.append(labels[permutation[s]])

	# repack the batches
	new_X = [[]]
	new_y = [[]]

	for b in range(len(shuff_samples)):
		new_X[-1].append(shuff_samples[b])
		new_y[-1].append(shuff_labels[b])
		if (len(new_X[-1]) >= batch_size):
			new_X.append([])
			new_y.append([])

	# trim out batches which are too small
	temp_X = []
	temp_y = []
	for batch in range(len(new_X)):
		if (len(new_X[batch]) == batch_size):
			temp_X.append(new_X[batch])
			temp_y.append(new_y[batch])
	new_X = temp_X
	new_y = temp_y

	return new_X, new_y


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		if (len(hidden_layers) == 1):
			self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
			self.fc2 = nn.Linear(hidden_layers[0], output_size)
			#self.fc2 = nn.Conv1d(hidden_layers[0],output_size,3,stride=2)
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
		elif (len(hidden_layers) == 4):
			self.fc1 = nn.Linear(input_size, hidden_layers[0]) # 2 Input noses, 50 in middle layers
			self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
			self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
			self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
			self.fc5 = nn.Linear(hidden_layers[3], output_size)
			self.rl1 = nn.ReLU()
			self.rl2 = nn.ReLU()
			self.rl3 = nn.ReLU()
			self.rl4 = nn.ReLU()
		
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
		elif (len(hidden_layers) == 4):
			x = self.fc1(x)
			x = self.rl1(x)
			x = self.fc2(x)
			x = self.rl2(x)
			x = self.fc3(x)
			x = self.rl3(x)
			x = self.fc4(x)
			x = self.rl4(x)
			x = self.fc5(x)

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
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	trainingdataX, trainingdataY, testdataX, testdataY = get_data()

	'''
	trainingdataX = trainingdataX.cuda()
	trainingdataY = trainingdataY.cuda()
	testdataX = testdataX.cuda()
	testdataY = testdataY.cuda()
	'''

	losses = []

	avg_running_loss = 0
	for epoch in range(NumEpoches):

		trainingdataX,trainingdataY = permute_data(trainingdataX,trainingdataY)

		running_loss = 0.0
		for i, data in enumerate(trainingdataX, 0):
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
				running_loss = 0.0
			
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
	h = "{}-".format(input_size)
	for hid in hidden_layers:
		h += "{}-".format(hid)
	h += "{}".format(output_size)
	print("NN: hid_layers: {} r2: {} pred[10]: {} actl[10]: {} train_size: {} test_size: {} epochs: {} batch_size: {} learn_rate: {}".format(h,r2,predictions[10],actual[10],training_set_size,test_set_size,NumEpoches,batch_size,learning_rate))

	'''
	predictions = []
	losses = []
	for i in range(len(trainingdataX)):
		test_sample = np.asarray(trainingdataX[i])
		test_label = np.asarray(trainingdataY[i])

		pred = (net(Variable(torch.FloatTensor(test_sample))))
		for z in range(len(pred)):
			p = pred[z].data.numpy()
			predictions.append(p)
		err = 0.0
		for p in range(len(pred)):
			err += pred[p] - (Variable(torch.FloatTensor(test_label[p])))
		err /= len(pred)
		e = err.data.numpy()
		losses.append(e)

	print((losses[0]))
	print(trainingdataY[0])

	
	print("predictions[0]: {}".format(predictions[0]))
	print("trainingdataY[0]: {}".format(trainingdataY[0]))
	errors = abs(predictions - trainingdataY)
	print("errors[0]: {}".format(errors[0]))
	sq_errors = errors ** 2
	sum_sq_errors = np.sum(sq_errors)
	res_errors = abs(trainingdataY - np.mean(trainingdataY)) ** 2
	sum_res_sq = np.sum(res_errors)
	r2 = 1 / (sum_sq_errors - sum_res_sq)
	print("r2: {}".format(r2))
	'''
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
	#plt.show()
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

	'''
	print("X contains: {} samples".format(len(X_)))
	print("Y contains: {} samples".format(len(y_)))
	print("X_val contains: {} samples".format(len(X_val)))
	print("Y_val contains: {} samples".format(len(y_val)))
	'''
	X = np.load("features_all_norm.npy")
	y = np.load("labels_all.npy")

	X = X[:,:13]

	X_ = X[:training_set_size]
	y_ = y[:training_set_size]
	X_val = X[-test_set_size:]
	y_val = y[-test_set_size:]


	clf = SVR(C=C, epsilon=epsilon,kernel='rbf')
	#samples = []
	#labels = []
	#for batch in range(len(trainingdataX)):
	#	for sample in range(len(trainingdataX[batch])):
	#		samples.append(trainingdataX[batch][sample])
	#		labels.append(trainingdataY[batch][sample][0])
	clf.fit(X_,y_)
		#print("X_train contains {} samples; y_train contains {} labels.".format(len(trainingdataX[batch]),len(trainingdataY[batch])))
		#print("X_test contains {} samples; y_test contains {} labels.".format(len(testdataX),len(testdataY)))
		#clf.fit(trainingdataX[batch],trainingdataY[batch])
		#predicted = clf.predict(testdataX[0])
		#errors = abs(predicted - testdataY[0])
		#print(clf.score(testdataX[0],testdataY[0]))
		#print("E: {}".format(np.mean(errors)))

	samples = []
	labels = []
	pred = []
	for sample in range(len(X_val)):
		#print(testdataX[batch][sample])
		samples.append(X_val[sample])
		labels.append(y_val[sample])
	pred = clf.predict(samples)
	errs = pred - labels

	print("SVM: R^2: {} C: {} eps: {}\n".format(clf.score(samples,labels),C,epsilon))
	
	
main()