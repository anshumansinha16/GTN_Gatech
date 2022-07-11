#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GTN(layers.Layer): # layers.Layer keeps track of everything under the hood!
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        
        self.layers = []

        for i in tf.range(num_layers):
            if i == 0:
                self.layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                self.layers.append(GTLayer(num_edge, num_channels, first=False))
        
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value= w_init(shape=(w_in, w_out)),trainable=True)
        
        b_init = tf.random_normal_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(w_out,)),trainable=True)
        
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.linear1 = tf.keras.layers.Dense( self.w_out, input_shape =(self.w_out*self.num_channels, ), activation= None) 
        self.linear2 = tf.keras.layers.Dense( self.num_class, input_shape=(self.w_out, ), activation= None)

        self.reset_parameters()

    def gcn_conv(self,X,H):
        
        X = tf.matmul(X, self.weight)
        H = self.norm(H, add=True)
        
        print('X:', X)
        print('H:', H)
        
        return tf.matmul(tf.transpose(H),X)

    
    def reset_parameters(self):
        
        initializer = tf.keras.initializers.GlorotUniform()
        values = initializer(shape=self.weight.shape)
        
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(2, 2))
        
        
    def normalization(self, H):
        
        for i in tf.range(self.num_channels):
            if i==0:
                
                H_ = tf.expand_dims(self.norm(H[i,:,:]), 0)
            else:
                
                H_ = tf.concat((H_, tf.expand_dims(self.norm(H[i,:,:]), 0)),0)
                
        return H_
    
    def norm(self, H, add=False):

        H = tf.transpose(H)
        if add == False:
            H = H*(tf.cast((tf.eye(H.shape[0])==0), tf.float32))
        else:
            H = H*(tf.cast((tf.eye(H.shape[0])==0), tf.float32)) + tf.cast((tf.eye(H.shape[0])==0), tf.float32)
        
        deg = tf.reduce_sum(H, 0).numpy()
        deg_inv = (1/deg)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*(tf.cast((tf.eye(H.shape[0])==0), tf.float32))
        
        H = tf.matmul(deg_inv,H)
        H = tf.transpose(H)
        return H
    
    def call(self, A, X, target_x, target):
        A = tf.expand_dims(A, 0)
        A = tf.transpose(A, perm=[0,3,1, 2])
        Ws = []
        
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A) #self.layers = nn.ModuleList(layers)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
            
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)

        for i in range(self.num_channels):
            if i==0:
                X_ = tf.nn.relu(self.gcn_conv(X,H[i])).numpy()
            else:
                X_tmp = tf.nn.relu(self.gcn_conv(X,H[i])).numpy()
                X_ = tf.concat((X_,X_tmp), 1)

        X_ = self.linear1(X_)
        X_ = tf.nn.relu(X_).numpy()
        y = self.linear2(X_[target_x])
        
        loss = self.loss(y, target)
        return loss, y, Ws
    
class GTLayer(keras.layers.Layer):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first

        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def call(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = tf.matmul( a, b)

            W = [tf.stop_gradient(tf.nn.softmax(self.conv1.weight, axis=1).numpy()),
                 tf.stop_gradient(tf.nn.softmax(self.conv2.weight, axis=1).numpy()) ]

        else:
            
            a = self.conv1(A)
            
            H = tf.matmul(H_, a)
            W = [tf.stop_gradient(tf.nn.softmax(self.conv1.weight, axis=1).numpy())]
        return H,W


class GTConv(keras.layers.Layer):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels        
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            initial_value=w_init(shape=(out_channels,in_channels,1,1)),
            trainable=True)
        self.bias = None
        self.scale = tf.Variable([0.1] , trainable=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        n = self.in_channels
        self.weight= tf.fill(self.weight.shape, 0.1 )
    
    def call(self, A):
        print('in')
        print(A.shape)
        print('sw',self.weight)
        A = tf.reduce_sum(A*(tf.nn.softmax(self.weight,1)), 1)
        print('lock')
        return A 

