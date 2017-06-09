res = open("res.dat",'r')
rs = []
sizes = []

for line in res:
	split = line.split(" ")
	r2 = (float)(split[4])
	size_1 = (int)(split[1][1:-1])
	size_2 = (int)(split[2][:-1])
	rs.append(r2)
	sizes.append([size_2])

import matplotlib.pyplot as plt 
plt.figure()
plt.plot(sizes,rs,'x',ms=2,mew=3)
plt.suptitle("R^2 for 4000 test samples with 1 hidden layer")
plt.xlabel("Size of hidden layer")
plt.ylabel("R^2 value")
plt.show()