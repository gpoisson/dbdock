import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from sklearn.svm import SVR


############################################################
### NEURAL NETWORK PARAMETERS
############################################################
input_size = 45						# size of input features

hidden_layers = [50]				# default hidden layer configuration

# user defined hidden layers
if (len(sys.argv) == 2):
	hidden_layers = [(int)(sys.argv[1])]															# one hidden layer
elif (len(sys.argv) == 3):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2]))]										# two hidden layers
elif (len(sys.argv) == 4):
	hidden_layers = [((int)(sys.argv[1])),((int)(sys.argv[2])),((int)(sys.argv[3]))]				# three hidden layers

output_size = 1						# size of output features
batch_size = 4						# number of samples per batch
training_set_size = 1000			# number of samples in training set
test_set_size = 2000				# number of samples in test set
epochs_count = 60					# number of times the network is repeatedly trained on the training data
learning_rate = 0.012				# speed at which the SGD algorithm proceeds in the opposite direction of the gradient

#############################################################
### SVM PARAMETERS
#############################################################
C = 100000.0
epsilon = 0.001

#############################################################
### OTHER PARAMETERS
#############################################################
permutation_noise = 0.00				# each time data is permuted, the data which is returned has some amount of
										#  noise added, proportionate to the 

#############################################################
### GLOBAL DATA PLACEHOLDERS -- DO NOT CHANGE
#############################################################
instance_permutation_order = []			# this is used to keep track of the order in which
										#  data is permuted, so that feature and label data
										#  can be permuted cohesively

# Returns a set of data for training and for testing, including feature and label data for each set
def get_data(lig_file,label_file,batch=True,subset="all_features"):
	global training_set_size, test_set_size
	ligands = np.load(lig_file)							# Reads the binary file containing ligand mol objects
	labels = np.load(label_file)						# Reads the binary file containing docking energy measurements

	if (subset == "first_order_only"):					# subset=1 --> only first 13 features (1st order features)	
		ligands = ligands[:,:13]
	elif (subset == "second_order_only"):				# subset=2 --> only last 20 features (2nd order)
		ligands = ligands[:,14:34]

	if (test_set_size + training_set_size > len(ligands)):
		print("Warning: Requested test set size ({}) and requested training set size ({}) requires {} samples.".format(test_set_size,training_set_size,(test_set_size+training_set_size)))
		print("         There are {} samples available. They will all be used, divided using the requested proportions.".format(len(ligands)))
		train_ratio = training_set_size / (float)(training_set_size + test_set_size)
		training_set_size = (int)(train_ratio * len(ligands))
		test_set_size = len(ligands) - training_set_size
		print("   New training set size:  {} samples".format(training_set_size))
		print("   New test set size:      {} samples".format(test_set_size))

	## Shuffle step
	ligands, labels = permute_data(ligands, labels, batched=False, add_noise=True)				# Randomly shuffle dataset prior to sampling

	ligands = np.asarray(ligands)
	labels = np.asarray(labels)
	
	train_ligands = ligands[:training_set_size]									# Divide data up into training and testing sets
	train_labels = labels[:training_set_size]
	test_ligands = ligands[-test_set_size:]
	test_labels = labels[-test_set_size:]

	##	Batching step
	trainingdataX = [[]]
	trainingdataY = [[]]

	for sample in range(len(train_ligands)):
		trainingdataX[-1].append(train_ligands[sample])
		trainingdataY[-1].append([(float)(labels[sample][1])])
		if ((len(trainingdataX[-1])) >= batch_size):
			trainingdataX.append([])
			trainingdataY.append([])

	temp_x = []
	temp_y = []
	
	for batch in range(len(trainingdataX)):										# trim out incomplete batches
		if (len(trainingdataX[batch]) == batch_size):
			temp_x.append(trainingdataX[batch])
			temp_y.append(trainingdataY[batch])
	
	trainingdataX = temp_x
	trainingdataY = temp_y

	testdataX = [[]]
	testdataY = [[]]

	for sample in range(len(test_ligands)):
		testdataX[-1].append(test_ligands[sample])
		testdataY[-1].append([(float)(test_labels[sample][1])])
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


	if (batch == False):														# If batching is not needed (SVM) then return unbatched data
		#print("TDX: {}".format(train_ligands))
		#print(len(train_ligands),len(train_ligands[0]))
		#print("TDY: {}".format(train_labels))
		#print(len(train_labels))
		temp_train_ligs = []
		temp_train_labs = []
		temp_test_ligs = []
		temp_test_labs = []
		for batch in trainingdataX:
			for entry in batch:
				temp_train_ligs.append(entry)
		trainingdataX = temp_train_ligs
		for batch in trainingdataY:
			for entry in batch:
				temp_train_labs.append(entry)
		trainingdataY = temp_train_labs
		for batch in testdataX:
			for entry in batch:
				temp_test_ligs.append(entry)
		testdataX = temp_test_ligs
		for batch in testdataY:
			for entry in batch:
				temp_test_labs.append(entry)
		testdataY = temp_test_labs
		return trainingdataX,trainingdataY,test_ligands,test_labels

	return trainingdataX, trainingdataY, testdataX, testdataY

def permute_data(X,y,batched=True,add_noise=False):
	global instance_permutation_order

	samples = []
	labels = []

	shuff_samples = []
	shuff_labels = []

	if (batched):
		# unpack the batches
		for batch in range(len(X)):
			for s in range(len(X[batch])):
				samples.append(X[batch][s])
				labels.append(y[batch][s])
	else:
		samples = X
		labels = y

	# define a fixed permutation
	if (len(instance_permutation_order) == 0):
		instance_permutation_order = np.random.permutation(len(samples))	# This gets populated with a list of values between [0 - (# samples)] and
																			#   is used to shuffle two sets of values cohesively (ligands and labels)
	if (len(instance_permutation_order) == len(samples)):					# If permuting all samples, enter here
		# shuffle the data
		for s in range(len(samples)):
			shuff_samples.append(samples[instance_permutation_order[s]])
			shuff_labels.append(labels[instance_permutation_order[s]])
	else:																	# If permuting a single batch, (neural network training epochs), enter here
		epoch_permutation_order = np.random.permutation(len(samples))
		for s in range(len(samples)):
			shuff_samples.append(samples[epoch_permutation_order[s]])
			shuff_labels.append(labels[epoch_permutation_order[s]])

	if (add_noise):
		# add random noise when first loading data to help prevent overfitting
		shuff_samples = np.asarray(shuff_samples)
		for samp in range(len(shuff_samples)):
			noise = np.random.rand() * permutation_noise
			sign = np.random.rand()
			if (sign < 0.5):													# Noise has 50% chance of being negative
				noise *= -1
			for feature in range(len(shuff_samples[samp])):
				feature_set = (shuff_samples[:,feature])
				median = np.median(feature_set)
				shuff_samples[samp,feature] += (median * noise)

	if (batched == False):
		return shuff_samples, shuff_labels

	

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

def train_SVM(trainingdataX,trainingdataY,testdataX,testdataY):
	clf = SVR(C=C, epsilon=epsilon,kernel='rbf')
	clf.fit(trainingdataX,(trainingdataY[:,1]))

	samples = []
	labels = []
	pred = []
	for sample in range(len(testdataX)):
		samples.append(testdataX[sample])
		labels.append((float)(testdataY[sample][1]))
	pred = clf.predict(samples)
	errs = pred - labels

	r2 = clf.score(samples,labels)
	print("SVM: R^2: {} C: {} eps: {}".format(r2,C,epsilon))
	return clf, r2

def train_NN(trainingdataX,trainingdataY,testdataX,testdataY):
	## Create Network
	net = Net()

	## Optimization and Loss
	criterion = nn.MSELoss()
	#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	losses = []
	avg_running_loss = 0
	for epoch in range(epochs_count):
		trainingdataX,trainingdataY = permute_data(trainingdataX,trainingdataY,add_noise=False)
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
				avg_running_loss = running_loss / epochs_count
				losses.append(avg_running_loss)
				running_loss = 0.0
			
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

	r2 = r2_score(actual,predictions)
	h = "{}-".format(input_size)
	for hid in hidden_layers:
		h += "{}-".format(hid)
	h += "{}".format(output_size)
	print("NN: hid_layers: {} r2: {} train_size: {} test_size: {} epochs: {} batch_size: {} learn_rate: {}".format(h,r2,training_set_size,test_set_size,epochs_count,batch_size,learning_rate))
	return net, r2

def train_and_test_svm_and_nn(ligand_file, label_file):
	#svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y = get_data(batch=False,subset="first_order_only")
	#nn_tr_X, nn_tr_y, nn_ts_X, nn_ts_y = get_data(batch=True,subset="first_order_only")
	svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y = get_data(ligand_file, label_file, batch=False,subset="all_features")
	nn_tr_X, nn_tr_y, nn_ts_X, nn_ts_y = get_data(ligand_file, label_file, batch=True,subset="all_features")
	print(len(svm_ts_y))
	print(len(nn_ts_y))
	print(len(nn_ts_y[0]))
	print(len(nn_ts_y[-1]))
	svm_model, r2_svm = train_SVM(svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y)
	nn_model, r2_nn = train_NN(nn_tr_X, nn_tr_y, nn_ts_X, nn_ts_y)
	svm_pred = svm_model.predict(svm_ts_X)
	nn_pred = []
	actual = []
	losses = []
	for batch in range(len(nn_ts_X)):
		batch_prediction = nn_model(Variable(torch.FloatTensor(nn_ts_X[batch])))
		batch_prediction = batch_prediction.data.numpy()
		for p in range(len(batch_prediction)):
			nn_pred.append(batch_prediction[p])
			actual.append(nn_ts_y[batch][p])
			losses.append(abs(nn_pred[-1] - nn_ts_y[batch][p]))

	plt.figure()
	print("\n{}\n{}\n".format(len(actual),len(svm_pred)))
	plt.plot(actual,svm_pred,'x',color='b',ms=2,mew=3)
	plt.plot(actual,actual,'x',color='r',ms=2,mew=3)
	plt.suptitle("SVM Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
	plt.ylabel("Predicted Binding Affinity")
	plt.xlabel("Known Binding Affinity")
	plt.grid(True)
	plt.show()

	plt.figure()
	plt.plot(actual,nn_pred,'x',color='g',ms=2,mew=3)
	plt.plot(actual,actual,'x',color='r',ms=2,mew=3)
	plt.suptitle("NN Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
	plt.ylabel("Predicted Binding Affinity")
	plt.xlabel("Known Binding Affinity")
	plt.grid(True)
	plt.show()

	plt.figure()
	plt.plot(actual,svm_pred,'x',color='b',ms=2,mew=3)
	plt.plot(actual,nn_pred,'x',color='g',ms=2,mew=3)
	plt.plot(actual,actual,'x',color='r',ms=2,mew=3)
	plt.suptitle("SVM vs NN Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
	plt.ylabel("Predicted Binding Affinity")
	plt.xlabel("Known Binding Affinity")
	plt.grid(True)
	plt.show()	

	return svm_model, r2_svm, nn_model, r2_nn
	