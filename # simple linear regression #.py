# simple linear regression #

# import relevan libraries

import numpy as np # for math opp
import matplotlib.pyplot as plt # for making graphs
from mpl_toolkits.mplot3d import Axes3D # provide ability to visualize 3d graph

# generate random input data to train

observation = 1000

x = np.random.uniform(low = -10,high = 10 ,size = (observation,1))
z = np.random.uniform(-10,10,(observation,1))

inputs = np.column_stack((x,z)) # to combine two vector became matrices/1d to 2d
print(inputs.shape)

# creating the targets we will aim at

noise = np.random.uniform(-1,1,(observation,1))

targets =  2*x - 3*z + 5 + noise  # the fuction is arbitrary

print(targets.shape)

#plot the training data

targets = targets.reshape(observation,)
x = x.reshape(observation,)
z = z.reshape(observation,)
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.plot(x,z,targets)
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('targets')
ax.view_init(azim = 100)
plt.show()
targets = targets.reshape(observation,1)
x = x.reshape(observation,1)
z = z.reshape(observation,1)

#initialize var

init_range = 0.1

weights = np.random.uniform(-init_range,init_range,size =(2,1))
bias = np.random.uniform(-init_range,init_range,size = 1)

print(weights)
print(bias)

# set a learning rate

learning_rate = 0.02

#train model

for i in range(300):
    outputs = np.dot(inputs,weights) + bias
    deltas = outputs - targets #deltas is diff betweeen predicted val and targets
    
    loss = np.sum(deltas**2) / 2 / observation
    
    print(loss) # if it decrease/dwindle our models works well
    
    deltas_scaled = deltas/observation
    weights = weights - learning_rate*np.dot(inputs.T,deltas_scaled) # we transpose the matrices so multuipication can occurr : alwyas check the dimension/shape of object
    bias = bias - learning_rate*np.sum(deltas_scaled)
    
print(weights,bias) # checking does the weight similar to our functions

plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()