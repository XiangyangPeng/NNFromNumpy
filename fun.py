# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:46:35 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import model_selection

data = np.linspace(-5, 5, 200, dtype=np.float)
target = np.cos(data)*3+np.sin(data)+data
# target = np.sin(data)
# target = np.ceil(data) % 2
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'], loss='mean_squared_error')
history = model.fit(x_train, y_train, batch_size=50, epochs=150, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
loss = history.history['loss']
val_loss = history.history['val_loss']
epoches = len(loss)
plt.plot(range(epoches), loss, marker='.', label='loss')
plt.plot(range(epoches), val_loss, marker='.', label='va_loss')
plt.show()
print('Test loss', score[0])
print('Test accuracy', score[1])
plt.scatter(x_test, y_test, c='blue')
plt.scatter(x_test, model.predict(x_test), c='red')
plt.show()
'''
test = np.linspace(1, 4, 30, dtype=np.float)
print(test)
pre = model.predict(test)
plt.plot(test, pre, 'r')
plt.plot(test, np.cos(test)*10, 'b')
plt.show()
'''