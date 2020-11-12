import numpy as np
import numpy.matlib
from scipy import linalg

def Lagrange(X,Y):
    n = X.size
    L = np.zeros((n,n))
    val0=0
    val=0
    i = 0
    while i < n:
        val0 = np.setdiff1d(X,X[i])
        val = ((1, -val0[0]))
        j = 1
        while j < n-1:
            val= np.convolve(val, [1,-val0[j]])
            j = j+1
        L[i,:] = val/np.polyval(val,X[i])
        i += 1
    print(L)
    #Mostrar Datos
    i = 0
    Coef = Y.dot(L)
    print(Coef)

X = np.array([-1, 0, 3, 4])
Y = np.array([15.5, 3, 8, 1])
Lagrange(X,Y)