import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf 

NEURON_NUMBER = 512
EPOCH_NUMBER = 500

dataset = pd.read_csv("cancer_data.csv")

x = dataset.drop(columns= ["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# split data into train and test (ratio 8:2)
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)
# 20& of data in testing set

# creating model
cancer_model = tf.keras.models.Sequential()

# neural network: input layer is x, output layer is y
# set up first layer of Neural Network
# .Dense() gives number of densely connected Neural Network layers
cancer_model.add(tf.keras.layers.Dense(NEURON_NUMBER, input_shape=x_train.shape[1:], activation="sigmoid" ))
# in this case gives 512 Neurons per Neural Network layer
cancer_model.add(tf.keras.layer.Dense(NEURON_NUMBER, activation="sigmoid"))
cancer_model.add(tf.keras.layer.Dense(NEURON_NUMBER, activation="sigmoid"))
cancer_model.add(tf.keras.layer.Dense(1, activation="sigmoid"))

#compiling model
cancer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model is based on improving accuracy of predicting new data

cancer_model.fit(x=x_train, y=y_train,
          batch_size=None, epochs=EPOCH_NUMBER,
          verbose=0, validation_data=None,
          steps_per_epoch=None, validation_steps=None,
          validation_batch_size=None, validation_freq=1)
#epochs -> number of times algorithm runs through data

#evaluate model using test data
cancer_model.evaluate(x_test, y_test, batch_size=None, verbose=2,
               sample_weight=None, steps=None, callbacks=None, max_queue_size=10)
