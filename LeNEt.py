#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:45:02 2018

@author: kumar
"""
import os
import struct
import path
from scipy import ndimage
from scipy.misc import imread
import numpy as np
from scipy import misc
f = misc.face()
misc.imsave('face.png',f) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(f)
plt.show()

face = misc.imread('face.png')
type(face)
face.shape,face.dtype
face.tofile('face.raw')
face_from_raw = np.fromfile('face.raw', dtype=np.uint8)
face_from_raw.shape
face_from_raw.shape = (768, 1024, 3)

for i in range(10):

    im = np.random.randint(0, 256, 10000).reshape((100, 100))

    misc.imsave('random_%02d.png' % i, im)
    
from glob import glob
filelist = glob('random*.png')   
filelist.sort()



train_X = os.path.join(path, 'train-images.idx3-ubyte')
train_Y = os.path.join(path, 'train-labels.idx1-ubyte')
from urllib import request
import gzip
import pickle

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")
    
def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()   
    
train_x,train_y,test_x,test_y=load()    
f.shape,f.dtype
def zero_pad(X, pad):
    
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    
    s = np.multiply(a_slice_prev, W) + b
    
    Z = np.sum(s)
   

    return Z


def ReLU(x):
    return x * (x > 0)

def conv_forward(A_prev, W, b, stride,pad):
 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape

    stride = stride
    pad = pad
    
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):                                 
        a_prev_pad = A_prev_pad[i]                     
        for h in range(n_H):                           
            for w in range(n_W):                      
                for c in range(n_C):                   
                   
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    Z[i, h, w, c] = ReLU(conv_single_step(a_slice_prev, W[...,c], b[...,c]))
                                        
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, W, b, stride,pad)
    
    return Z, cache


def pool_forward(A_prev, f,stride):
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = f
    stride = stride
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
   
    for i in range(m):                           
        for h in range(n_H):                     
            for w in range(n_W):                 
                for c in range (n_C):            
                    
                  
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    A[i, h, w, c] = np.max(a_prev_slice)
                    
    cache = (A_prev, f,stride)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


def initialize_conv_parameters(n_H,n_C,n_f):
    
    np.random.seed(0)
    
    W = np.random.randn(n_H, n_H,n_C,n_f) * 0.01
    b = np.zeros(shape=(1,1,1,n_f))
    
    
    assert(W.shape == (n_H,n_H,n_C,n_f))
    assert(b.shape == (1,1,1,n_f))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

from PIL import Image
from numpy import array
img = Image.open("cats.png")
arr = array(img)
    
W1=initialize_conv_parameters(3,3,10)['W']
b1=initialize_conv_parameters(3,3,10)['b']
W2=initialize_conv_parameters(5,10,20)['W']
b2=initialize_conv_parameters(5,10,20)['b']
 
A1,cache1=conv_forward(train_x, W1, b1, 2,1)
A1_pool,cache1_pool=pool_forward(A1, 1,2)
A2,cache2=conv_forward(A1_pool, W2, b2, 2,1)
A2_pool,cache2_pool=pool_forward(A2, 1,2)

def initialize_parameters(n_x, n_h, n_y):
    
    
    np.random.seed(0)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b
    
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL, Y):
   
    m = Y.shape[1]

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    
    dW = np.dot(dZ, cache[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (isinstance(db, float))
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        
        
    elif activation == "sigmoid":
       
        dZ = sigmoid_backward(dA, activation_cache)
        
    
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
   
    dAL = dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, 
                                                                                                        current_cache[1]), 
                                                                                       current_cache[0])
    
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, caches[1]), caches[0])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    
    L = len(parameters) // 2 

    
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        
    return parameters


final_conv=A2_pool.flatten()
W3=initialize_parameters(784,200,10)['W1']
b3=initialize_parameters(784,200,10)['b1']
W4=initialize_parameters(784,200,10)['W2']
b4=initialize_parameters(784,200,10)['b2']
A1,cache1=linear_activation_forward(final_conv, W3, b3,"relu")
A2,cache2=linear_activation_forward(A1, W4, b4,"sigmoid")
 cost=compute_cost(A2,train_Y)
while cost>= threshold :
    gradient=L_model_backward(A2, train_Y, ['cache1','cache2'])
    para=update_parameters(parameters, gradient,0.01)
    W3=para['W1']
    b3=para['b1']
    W4=para['W2']
    b4=para['b2']
    A1,cache1=linear_activation_forward(final_conv, W3, b3,"relu")
    A2,cache2=linear_activation_forward(A1, W4, b4,"sigmoid")
    cost=compute_cost(A2,train_Y)
    
    
    



A1,cache1=linear_activation_forward(train_x, W3, b3,"relu")
A2,cache2=linear_activation_forward(A1, W4, b4,"sigmoid")







