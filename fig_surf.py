#!/usr/bin/env python3

import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = np.loadtxt('1')

Nrow = a.shape[0]
N = int(np.sqrt(Nrow))


X = a[:,0].reshape([N,N])
Y = a[:,1].reshape([N,N])

E0 = a[:,2].reshape([N,N])
E1 = a[:,3].reshape([N,N])

ax.plot_surface(X,Y,E0)
ax.plot_surface(X,Y,E1)

ax.set_xlabel('x')
ax.set_ylabel('y') 
ax.set_zlabel('Energy')
ax.set_title('Potential Adiabats')

#ax.legend(['E0', 'E1'])


plt.show()
