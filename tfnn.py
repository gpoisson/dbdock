#  Fit a straight line, of the form y=m*x+b

import tensorflow as tf
import numpy as np

'''
Your dataset.
'''
#xs = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00] # Features
#ys = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00] # Labels

xs = np.load("features.npy")
ys = np.load("labels.npy")

'''
Initial guesses, which will be refined by TensorFlow.
'''
m_initial = -0.5 # Initial guesses
b_initial =  1.0

'''
Define free variables to be solved.
'''
m = tf.Variable(m_initial) # Parameters
b = tf.Variable(b_initial)

'''
Define the error between the data and the model one point at a time (slow).
'''
total_error = 0.0
for i in range(len(xs)):
    y_model = m*xs[i]+b # Output of the model aka yhat
    total_error += (ys[i]-y_model)**2 # Difference squared - this is the "cost" to be minimized

'''
Once cost function is defined, create gradient descent optimizer.
'''
optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error) # Does one step

'''
Something you just have to do.
'''
initializer_operation = tf.global_variables_initializer()

'''
All calculations are done in a session.
'''
with tf.Session() as session:

    session.run(initializer_operation)

    _EPOCHS = 10000 # number of "sweeps" across data
    for iteration in range(_EPOCHS):
        session.run(optimizer_operation)

    print('Slope:', m.eval(), 'Intercept:', b.eval())
    print('Total Error:', total_error.eval())