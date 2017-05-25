import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os

print(keras.__version__)
print(os.path.dirname(keras.__file__))
# fix random seed for reproducibility
np.random.seed(7)


# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

print(X.shape, X.dtype)
print(Y.shape, Y.dtype)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=10, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print()
print(model.metrics_names)
print(scores)
print()
for i, (name, score) in enumerate(zip(model.metrics_names, scores)):
    print("%2d: %10s: %.2f%%" % (i, name, score * 100))
