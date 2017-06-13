res = open("res.dat",'r')
smv_rs = []
nn_rs = []
avg_rs = []
sizes = []

for line in res:
	split = line.split(" ")
	if (split[0] == "SVM:"):
		r2 = (float)(split[2])
		smv_rs.append(r2)
	elif (split[0] == "NN:"):
		r2 = (float)(split[4])
		size = (int)(split[6])
		sizes.append(size)
		nn_rs.append(r2)
	else:
		r2 = (float)(split[1])
		avg_rs.append(r2)

import matplotlib.pyplot as plt 
plt.figure()
plt.plot(sizes,avg_rs,color='r',ms=2,mew=3)
plt.plot(sizes,smv_rs,color='b',ms=2,mew=3)
plt.plot(sizes,nn_rs,color='g',ms=2,mew=3)
plt.suptitle("R^2 for 2000 test samples\nSVM vs Neural Network")
plt.xlabel("Size of training set")
plt.ylabel("R^2 value")
plt.grid(True)
plt.show()