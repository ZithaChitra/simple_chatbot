# import numpy as np
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, LSTM


# #prepare the sequence
# length = 5
# seq = np.array([ i / float(length) for i in range(length) ])
# X = seq.reshape(1, length, 1)
# y = seq.reshape(1, length)
# print(seq.shape)


# # define lstm configuration
# n_neurons = length
# n_batch = 1
# epochs = 500

# # create lstm model
# model = keras.Sequential()
# model.add(layers.LSTM(n_neurons, input_shape=(5, 1)))
# model.add(layers.Dense(1))
# model.compile(loss="mean_squared_error", optimizer="adam")
# print(model.summary())

# # train lstm
# model.fit(X, y, epochs=epochs, batch_size=n_batch, verbose=2)

# # result
# result = model.predict(X, batch_size=n_batch, verbose=0)
# for value in result[0,:]:
# 	print("%.1f" % value)


# create LSTM
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)