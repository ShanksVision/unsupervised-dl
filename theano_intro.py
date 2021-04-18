#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:42:56 2020

@author: sjagadee
"""

import theano
import theano.tensor as T
import numpy as np

#Create theano variables
s = T.scalar('s')
v = T.vector('v')
A = T.matrix('A')

w= A.dot(v)

#Create the function 
matrix_dot_vec = theano.function(inputs=[A,v], outputs=w)

#Pass values and calculate result
A_val = np.array([[1,2,3], [4,5,6], [7,8,9]])
v_val = np.array([1,1,1])
w_val = matrix_dot_vec(A_val, v_val)

print(w_val)


#Lets do some smple grad update on cost function to find minima

wx = theano.shared(20.0, 'wx')
cost = wx*wx + wx + 5
weight_update = wx - 0.1*T.grad(cost, wx)
train = theano.function(inputs=[], outputs=cost, updates=[(wx, 
                                                          weight_update)])


for i in range(30):
    cost_val = train()
    print(cost_val)
    
print('Value of input at minima', wx.get_value())
