# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:01:32 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import random
'''
-----name rules-----------------
variable name:inputLayers
function name:initNet
-----assumption-----------------
all dense layer
activation function-  Leaky-ReLU
-----Function-------------------
BackPropagation
    args:net, target function, forward prediction matrix
    operation:change weight matrix of net

add layer (init as ones)
    args: input-vector(length = m);number of cells-n;previous net-tuple list
    operation: add a tuple(matrix)
    return: new net

optimizer
    args:sample input, sample output, iterations, train protocol
    operation:forward prediction-matrix(output of every layer)       
              BackPropagation-change weights of net
              in circulation, change weights in protocol

test
    args:sample test, net
    return:prediction              
                  

-----DataStructure--------------

net-a sequence of layer

layer-w(w_kj, j = 0, 1, 2, ..., ), b(w_k0), f, label
     -input hidden outut
 matrix((m+1)*n) 


target function
               -MSE cross entropy
'''               


def backpropagation(net, samplePre, sampleTar, eta):
    """
    :param net:A list of dictionary-[{'w':np.array(), 'actifun':string, ('a':float)}, {}, ...] 'w':neuron_n(k)*(neuton(k-1)+1)
    :param samplePre:Prediction of sampleIn
    :param sampleTar:Target of sampleIn
    :param eta: learning rate
    :return:dweight
    """
    layer_n = len(net)
    sigma = []
    dweight = []
    for layer_i in range(layer_n-1, 0, -1):
        layer_y = samplePre[layer_i]['y']
        if net[layer_i]['actifun'] == 'sigmoid':
            layer_fy = 2/(np.exp(layer_y)+np.exp(layer_y*-1)+2)
        elif net[layer_i]['actifun'] == 'LeakyReLU':
            layer_a = net[layer_i]['LRa']
            layer_fy = layer_y.copy()
            layer_fy[np.where(layer_y < 0)] = layer_a
            layer_fy[np.where(layer_y >= 0)] = 1
        elif net[layer_i]['actifun'] == 'linear':
            layer_fy = 1
        if layer_i == layer_n-1:
            sigma_i = (sampleTar-samplePre[layer_i]['z'])*layer_fy
        else:
            layer_w = net[layer_i+1]['w'][:, 1:]
            sigma_i = layer_w.T.dot(sigma[layer_n-layer_i-2])*layer_fy
        sigma.append(sigma_i)
        layer_zn = samplePre[layer_i-1]['z']
        dweight_i = (sigma_i.reshape(-1, 1))*layer_zn*eta
        dweight.append(dweight_i)
    dweight.reverse()
    return dweight  # not include input layer

        
def addLayer(net, neuron_n, init_w, actifun, LRa = 0.1):
    """
    :param net:A list of dictionary-[{'w':np.array(), 'actifun':string, ('a':float)}, {}, ...] 'w':neuron_n(k)*(neuton(k-1)+1)
    :param neuron_n:Number of neurons
    :param init_w:Initial value of weight matrix
    :param actifun:Activation function-"linear" "sigmoid" "ReLU" "LeakyReLU"
    :param LRa:Value of a in activation function Leaky-ReLU
    :return:
    """
    # this function can only add layer to a net with at least one layer(input layer)
    _neuron_n = net[-1]['w'].shape
    weight = np.ones([neuron_n, _neuron_n[0]+1], np.float)*init_w
    layer = {'w': weight, 'actifun': actifun}
    if actifun == "LeakyReLU":
        layer['LRa'] = LRa
    net.append(layer)


def predict(net, sampleIn, mode):
    """
    :param net:A list of dictionary-[{'w':np.array(), 'actifun':string, ('a':float)}, {}, ...]  'w':neuron_n(k)*(neuton(k-1)+1)
    :param sampleIn:a sample or a list of samples;np.array
    :param mode: predict mode-'test' 'train'
    :return:pre:a list of a list of prediction
    """
    if mode == 'train':
        sampleNumber = 1
        sampleIn = np.expand_dims(sampleIn, axis=0)
    elif mode == 'test':
        sampleNumber = sampleIn.shape[0]
    pre = []
    layer_n = len(net)
    for sampleIndex in range(sampleNumber):
        samplePre = []
        _sampleIn = np.append(sampleIn[sampleIndex], 1)
        layerPre = {'y': sampleIn[sampleIndex], 'z': _sampleIn}
        samplePre.append(layerPre)
        for layer_i in range(1, layer_n):
            # y
            layer_x = samplePre[layer_i-1]['z']
            layer_wb = net[layer_i]['w']
            layer_y = layer_wb.dot(layer_x)
            # z
            layer_z = layer_y.copy()
            if net[layer_i]['actifun'] == 'sigmoid':
                layer_z = (1-np.exp(layer_z*-1))/(1+np.exp(layer_z*-1))
            elif net[layer_i]['actifun'] == 'LeakyReLU':
                layer_a = net[layer_i]['LRa']
                index = np.where(layer_z<0)
                layer_z[index] = layer_z[index]*layer_a
            elif net[layer_i]['actifun'] == 'linear':
                layer_z = layer_y
            # 除输出层外，增加一个输出单元为偏置单元
            if layer_i < layer_n-1:
                layer_z = np.append(layer_z, 1)
            layerPre = {'y':layer_y, 'z':layer_z}
            samplePre.append(layerPre)
        pre.append(samplePre)
    if mode == 'train':
        return samplePre
    else:
        return pre
    
def optimizer(net, _sample, epoches, batch_size, eta):
    """
    :param net: A list of dictionary-[{'w':np.array(), 'actifun':string, ('a':float)}, {}, ...] 'w':neuron_n(k)*(neuton(k-1)+1)
    :param _sample: [sampleIn, sampleTar]
    :param epoches:
    :param batch_size:
    :param eta: learning rate
    :return: a list of mse in learning process
    """
    layer_n = len(net)
    sample_n = len(_sample[0])
    mse = []
    for epoch_i in range(epoches):
        #shuffle the sampleIn and sampleTar to achieve random split
        indexes = np.array(range(sample_n))
        random.shuffle(indexes)
        sample = [_sample[0][indexes], _sample[1][indexes]]
        
        for batch_i in range(int(sample_n/batch_size)):
            mse_i = 0
            for train_i in range(batch_size):
                sample_i = batch_size*batch_i+train_i                
                sampleIn = sample[0][sample_i]
                sampleTar = sample[1][sample_i]
                samplePre = predict(net, sampleIn, 'train')
                mse_i = np.power((samplePre[-1]['z']-sampleTar), 2)+mse_i
                tempdweight = backPropagation(net , samplePre, sampleTar, eta)
                if train_i == 0:
                    dweight = tempdweight
                else:
                    for layer_i in range(layer_n-1):
                        dweight[layer_i] = dweight[layer_i]+tempdweight[layer_i]
            mse.append(mse_i/batch_size)
            for layer_i in range(layer_n-1):
                net[layer_i+1]['w'] = dweight[layer_i]+net[layer_i+1]['w']
    return mse
        
def initNet(sampleInDim):
    net = []
    #in function add layer, the number of neurons in the upper layer is needed
    weight = np.ones([sampleInDim, sampleInDim], np.float)
    inputLayer = {'w':weight}
    net.append(inputLayer)
    return net
    
'''
如果要拟合一个非线性的函数，Leaky-ReLU并不是一个好的选择。
训练时样本应该随机分组，如果按照固定的顺序来，可能会造成训练效果的上下波动
在回归问题中，激活函数必须谨慎选择，要保证输出结果的取值范围与所要拟合的数据的范围相当
'''            
if __name__ == '__main__':
    data_dim = 1
#    net = {'w':[], 'a':[], 'acti':[]}
#    #输入层
#    net['w'].append(np.ones([data_dim, data_dim], np.float))
#    net['a'].append(0.1)
    net = initNet(data_dim)
    #隐藏层 权值的初始化不能太大，否则会无法收敛
    addLayer(net = net, neuron_n = 10, init_w = 0.1, actifun = 'sigmoid')
    addLayer(net = net, neuron_n = 10, init_w = 0.1, actifun = 'sigmoid')
#    addLayer(net, 5, 0.1, 0.3)
#    addLayer(net, 10, 0.1, 0.2)
#    addLayer(net, 5, 0.1, 0.1)
#    addLayer(net, 3, 0.1, 0.1)
#    addLayer(net, 5, 0.1, 0.1)
#    addLayer(net, 3, 0.1, 0.1)
    #输出层
    addLayer(net = net, neuron_n = 1, init_w = 1, actifun = 'linear')
    #addLayer(net, 1, 0.1, 0.3)
    
    #数据生成, 3., 4., 6., 7., 9.
    '''
    sampleIn = np.array([-1., 2., 4., 6., 3., 7., 9.])
    sampleTar = np.array([-1., 2., 4., 6., 3., 7., 9.])
    sample = [sampleIn, sampleTar]
    '''
    sampleIn = np.linspace(0, 20, num = 500)
    #sampleTar = np.power(sampleIn, 2)
    #sampleTar = sampleIn
    sampleTar = np.sin(sampleIn)*10
    sample = [sampleIn, sampleTar]
    
    #训练
    epoches = 50
    batch_size = 20
    eta = 0.001#不当的学习率会导致无法收敛，试着把它调大一点就会发现这个情况
    mse = optimizer(net, sample, epoches, batch_size, eta)
    plt.plot(mse)
    plt.show()
    print("last trainmse:", mse[-1])
    #测试
    testnum = 50
    sampleTestIn = np.linspace(1, 5.3, num = testnum)
    #sampleTestTar = np.power(sampleTestIn, 2)
    #sampleTestTar = sampleTestIn
    sampleTestTar = np.sin(sampleTestIn)*10
    samplePreTar = []
    testmse = 0
    pre = predict(net, sampleTestIn, 'test')
    for test_i in range(testnum):        
        testmse = testmse+np.power((pre[test_i][-1]['z']-sampleTestTar[test_i]), 2)
        samplePreTar.append(pre[test_i][-1]['z'][0])
    testmse /= testnum
    #samplePreTar = np.array(samplePreTar)
    print("test MSE:", testmse)
    
    plt.plot(sampleTestIn, sampleTestTar)
    plt.plot(sampleTestIn, samplePreTar)    
    