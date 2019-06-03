# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:01:32 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
'''
-----assumption-----------------
all dense layer
activation function-  Leaky-ReLU
-----Function-------------------
BackPropagation
    args:net,target function,forward prediction matrix
    operation:change weight matrix of net

add layer (init as ones)
    args: input-vector(length=m);number of cells-n;previous net-tuple list
    operation: add a tuple(matrix)
    return: new net

optimizer
    args:sample input,sample output,iterations,train protocol
    operation:forward prediction-matrix(output of every layer)       
              BackPropagation-change weights of net
              in circulation,change weights in protocol

test
    args:sample test,net
    return:prediction              
                  

-----DataStructure--------------

net-a sequence of layer

layer-w(w_kj,j=0,1,2,...,),b(w_k0),f,label
     -input hidden outut
 matrix((m+1)*n) 


target function
               -MSE cross entropy
'''               

def backPropagation(net,pre,sampleTar,eta):
    layer_n=len(net['w'])
    sigma=[]
    dweight=[]
    for layer_i in range(layer_n-1,0,-1):
        layer_y=pre['y'][layer_i]
        layer_a=net['a'][layer_i]
        layer_fy=layer_y.copy()
        layer_fy[np.where(layer_y<0)]=layer_a
        layer_fy[np.where(layer_y>=0)]=1
        #计算 \sigma
        if layer_i==layer_n-1:
            sigma_i=(sampleTar-pre['z'][layer_i])*layer_fy
        else:
            layer_w=net['w'][layer_i+1][:,1:]
            sigma_i=(layer_w.T).dot(sigma[layer_n-layer_i-2])*layer_fy            
        sigma.append(sigma_i)
        #计算权值调整量
        layer_zn=pre['z'][layer_i-1]
        dweight_i=(sigma_i.reshape(-1,1))*layer_zn*eta
        
        dweight.append(dweight_i)
    dweight.reverse()
    #输入层->输出层；每一层最后一个作为偏置单元
    return dweight
    
def addLayer(net,cell_n,a,init_v):
    '''
    net: layer_n*cell_n*(cell_(n-1)+1),输入层全为0,偏置单元看作输出为1 的上一层的单元
    cell_n: cell numbers of new layer,dense defaulted
    a:Leaky_ReLU-a
    '''
    pcell_n=net['w'][-1].shape
    layer=np.ones([cell_n,pcell_n[0]+1],np.float)*init_v
    net['w'].append(layer)
    net['a'].append(a)
    
def predict(net,sampleIn):
    layer_n=len(net['w'])
    pre={'y':[],'z':[]}
    pre['y'].append(sampleIn)
    sampleIn=np.append(sampleIn,1)
    pre['z'].append(sampleIn)
    for layer_i in range(1,layer_n):
        #y
        layer_x=pre['z'][layer_i-1]
        #layer_w=net['w'][layer_i][:,1:]
        #layer_b=net['w'][layer_i][:,0]
        layer_wb=net['w'][layer_i]
        layer_y=layer_wb.dot(layer_x)
        
        #z
        layer_z=layer_y.copy()
        layer_a=net['a'][layer_i]
        index=np.where(layer_z<0)
        layer_z[index]=layer_z[index]*layer_a
        
        #除输出层外，增加一个单元
        if(layer_i<layer_n-1):
            #layer_y=np.append(layer_y,1)
            layer_z=np.append(layer_z,1)
        pre['y'].append(layer_y)
        pre['z'].append(layer_z)
    return pre
        
    
def optimizer(net,sample,epoches,batch_size,eta):
    layer_n=len(net['w'])
    sample_n=len(sample[0])
    mse=[]
    for epoch_i in range(epoches):
        for batch_i in range(int(sample_n/batch_size)):
            dweight=net['w'][1:]
            mse_i=0
            for train_i in range(batch_size):
                sample_i=batch_size*batch_i+train_i
                sampleIn=sample[0][sample_i]
                sampleTar=sample[1][sample_i]
                #前向预测+mse
                pre=predict(net,sampleIn)
                mse_i=np.power((pre['z'][-1]-sampleTar),2)+mse_i
                #print(pre['z'][-1])
                tempdweight=backPropagation(net,pre,sampleTar,eta)
                #累加
                for layer_i in range(layer_n-1):
                    dweight[layer_i]=dweight[layer_i]+tempdweight[layer_i]
            #调整权值
            #print(net['w'])
            mse.append(mse_i/batch_size)
            for layer_i in range(layer_n-1):
                net['w'][1:]=dweight
    return mse
        
            
    
            
if __name__=='__main__':
    data_dim=1
    net={'w':[],'a':[]}
    #输入层
    net['w'].append(np.ones([data_dim,data_dim],np.float))
    net['a'].append(0.1)
    #隐藏层 权值的初始化不能太大，否则会无法收敛
    addLayer(net,3,0.1,0.3)
    addLayer(net,3,0.1,0.2)
    addLayer(net,10,0.1,0.1)
    addLayer(net,10,0.1,0.1)
    addLayer(net,5,0.1,0.1)
    addLayer(net,3,0.1,0.1)
    #输出层
    addLayer(net,1,0.1,0.3)
    
    #数据生成,3.,4.,6.,7.,9.
    '''
    sampleIn=np.array([-1.,2.,4.,6.,3.,7.,9.])
    sampleTar=np.array([-1.,2.,4.,6.,3.,7.,9.])
    sample=[sampleIn,sampleTar]
    '''
    sampleIn=np.linspace(0,10,num=1000)
    sampleTar=np.power(sampleIn,2)
    sample=[sampleIn,sampleTar]
    
    #训练
    epoches=500
    batch_size=20
    eta=0.000001#不当的学习率会导致无法收敛，试着把它调大一点就会发现这个情况
    mse=optimizer(net,sample,epoches,batch_size,eta)
    plt.plot(mse)
    plt.show()
    print("last trainmse:",mse[-1])
    #测试
    testnum=50
    sampleTestIn=np.linspace(2.6,8.2,num=testnum)
    sampleTestTar=np.power(sampleTestIn,2)
    testmse=0
    for test_i in range(testnum):
        pre=predict(net,sampleTestIn[test_i])
        testmse=testmse+np.power((pre['z'][-1]-sampleTestTar[test_i]),2)
    testmse=testmse/testnum
    print("test MSE:",testmse)
        