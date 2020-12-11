import numpy as np

a = np.load('a.npy')
b = np.load('b.npy')
c = []

c = np.append(a, b)

np.save('merge.npy', c)
