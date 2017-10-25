'''
Created on 30 Sep 2017

@author: dai
'''

"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

import numpy as np
from miniflow import *
#1. Add and Mul
# x, y, z = Input(), Input(), Input()
# 
# f = Add(x, y, z)
# t = Mul(x,y,z)
# 
# feed_dict = {x: 4, y: 5, z: 10}
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
# out_mul=forward_pass(t,graph)
# 
# # should output 19
# print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
# print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], out_mul))

# #2. Linear
# inputs, weights, bias = Input(), Input(), Input()
# 
# f = Linear(inputs, weights, bias)
# 
# feed_dict = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
# print(output)
# print(6*0.5+14*0.25+3*1.4+2)


# #3. Linear -> Sigmoid
# X, W, b = Input(), Input(), Input()
# 
# f = Linear(X, W, b)
# g = Sigmoid(f)
# 
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
# 
# feed_dict = {X: X_, W: W_, b: b_}
# 
# graph = topological_sort(feed_dict)
# output = forward_pass(g, graph)
# 
# """
# Output should be:
# [[  1.23394576e-04   9.82013790e-01]
#  [  1.23394576e-04   9.82013790e-01]]
# """
# print(output)

#4. MSE
import numpy as np
from miniflow import *

y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
forward_pass(graph)

"""
Expected output

23.4166666667
"""
print(cost.value)