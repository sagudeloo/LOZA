import numpy as np
import numpy.matlib

matrix=[[2, -1, 0, 3],
        [1, 0.5, 3, 8],
        [14, 5, -2, 3],
        [0, 13, -2, 11]]
vector=[[1],
        [1],
        [1],
        [1]]

def gaussSpl(Ma, b):
    #Getting matrix dimention
    n = len(Ma)
    #Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))
    for i in range(n-1):
        for j in range(i+1,n):
            if (M[j,i] != 0):
                M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]
    return backSust(M)

def backSust(M):
    #Getting  matrix dimention
    n = len(M)
    #Initializing a zero vector
    x = np.matlib.zeros((n,1))
    x[n-1]=M[n-1,n]/M[n-1,n-1]
    for i in range(n-2, -1, -1):
        aux1 = np.hstack((1, np.asarray(x[i+1:n]).reshape(-1)))
        aux2 = np.hstack((M[i,n], np.asarray(-M[i,i+1:n]).reshape(-1)))
        x[i] = np.dot(aux1,aux2)/M[i,i]
    return x


gaussSpl(matrix, vector)