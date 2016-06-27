import numpy as np
import matplotlib.pyplot as plt

def makeSinglePlot(x_data,y_data,title='Plot',x_label='x',y_label='y',axes_on=True,marker_type='x',add_lobf=True,x_min=None,x_max=None,y_min=None,y_max=None,axis_equal=False):
	plt.figure()																	# make a plot figure
	plt.plot(x_data,y_data,marker_type)												# add the data to the plot
	if add_lobf:																	# add a line of best fit
		plt.plot(x_data, np.poly1d(np.polyfit(x_data, y_data, 1))(x_data))
	plt.suptitle(title)																# add plot title, labels
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if (x_min != None):																# fix boundaries of the plot
		plt.xlim([x_min,x_max])
	if (y_min != None):
		plt.ylim([y_min,y_max])
	if axes_on:																		# enable grid axes
		plt.grid(True)
	if axis_equal:
		plt.axis('equal')
	plt.show()

# Produce multiple plots on one firgure
# 
#def makeMultiPlot(x_data,y_data,title='Plot',x_label='x',y_label='y',axes_on=True,marker_type='x',add_lobf=True,x_min=None,x_max=None,y_min=None,y_max=None):