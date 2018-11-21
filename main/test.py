import os
import util
import numpy as np
import math
RANDOM_SEED = 1
# np.random.seed(RANDOM_SEED)

def increment(i):
    return i+1

def decrement(i):
    return i-1

def log(i):
    return math.log(i, 10)

operation = [increment, decrement, log]

op = np.random.randint(2, size=(10))

print(op)

a = np.ones(10) * 10

print(a)

b = []

for i in range(10):
    b.append(operation[op[i]](a[i]))

print(b)

c = np.asarray([[1,2,3],[4,5,6]])
d  = [[1,2,3],[4,5,6]]

print(np.sum(d, 0))
