import numpy as np 
import tensorflow as tf
from sklearn import preprocessing

#Data 

npz = np.load('D:/aisy data science/ml/Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float64)
train_targets = npz['targets'].astype(np.int64)

npz = np.load('D:/aisy data science/ml/Audiobooks_data_validation.npz')
validation_inputs,validation_targets = npz['inputs'].astype(np.float64),npz['targets'].astype(np.int64)

npz = np.load('D:/aisy data science/ml/Audiobooks_data_test.npz')
test_inputs,test_targets = npz['inputs'].astype(np.float64),npz['targets'].astype(np.int64)


# model 

input_size = 10
output_size = 2
hidden_layer = 50 

model= tf.keras.Sequential([
                             tf.keras.layers.Dense(hidden_layer,activation='relu'),
                             tf.keras.layers.Dense(hidden_layer,activation='relu'),
                             tf.keras.layers.Dense(output_size,activation='softmax')                 
])

model.compile(optimizer='adam',loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])

batch_size = 100
max_epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2) # making sure taht increase of loss_val  2 times before stop-

model.fit(train_inputs,
          train_targets,batch_size = batch_size,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data= (validation_inputs,validation_targets),
          verbose=2)
new_inputs_unscaled = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
new_inputs_scaled = preprocessing.scale(new_inputs_unscaled)
new_inputs_scaled = new_inputs_scaled.reshape(1,-1)
 
prediction_indexes = model.predict(new_inputs_scaled)
print(np.argmax(prediction_indexes, axis=-1))


