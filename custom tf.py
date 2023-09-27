import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#data generation

observation = 1000

x = np.random.uniform(-10,10,size = (observation,1))
z = np.random.uniform(-10,10,size = (observation,1))

generated_inputs = np.column_stack((x,z))

noises = np.random.uniform(-1,1,size = (observation,1))
 
generated_targets = 2*x - 3*z + 5 + noises

np.savez('TF_intro',inputs = generated_inputs,targets = generated_targets)

# Solving Using Tensor Flow

training_data = np.load('TF_intro.npz')


input_size = 2 # x and z
output_size = 1  # targets value or y

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(output_size,
                                                kernel_initializer=tf.random_uniform_initializer(minval = -0.1,maxval = 0.1), # kernel mean weight
                                                bias_initializer = tf.random_uniform_initializer(minval = -0.1,maxval = 0.1) 
                                                )
                            ])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.02)
model.compile(optimizer='sgd',loss= 'mean_squared_error') # sgd mean Stophicated gradient descent

model.fit(training_data['inputs'],training_data['targets'],epochs=100,verbose=1)

# Extract weights and bias

print(model.layers[0].get_weights()) # to get coeeff for weights and bias 

weight = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

#predicting value

print(model.predict_on_batch(training_data['inputs']).round(1),'\n')

print(training_data['targets'].round(1))

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])),np.squeeze(training_data['targets']))
plt.xlabel('Predicted Value')
plt.ylabel('Targets')
plt.show()


