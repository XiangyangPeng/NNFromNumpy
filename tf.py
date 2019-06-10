# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:47:56 2019

@author: Administrator
"""
import numpy as np
import tensorflow.keras as keras

data=np.linspace(0,10,num=500)
targets=np.sin(data)
model=keras.Sequential([
        keras.layers.Dense(10,activation='sigmoid',input_shape=(1,)),
        keras.layers.Dense(1,activation='linear')
])
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model.fit(data,targets,batch_size=32,epochs=5)

test=np.linspace(2.3,5.4,num=50)
testT=np.sin(test)
preT=model.predict(test,batch_size=5)