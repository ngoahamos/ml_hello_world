import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([100.0]))

print("rule for this data is 2{} - 1".format('x'))
weights = l0.get_weights()
x = weights[0][0][0]
b = weights[1][0]
print("the model learned {}x {}".format(x,b))
