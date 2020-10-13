import numpy as np
import numpy.matlib


def gaussSimple(Ma, b):
    # Getting matrix dimention
    n = len(Ma)

    # Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))

    # Matrix reduction
    for i in range(n-1):
        for j in range(i+1,n):
            if (M[j,i] != 0):
                M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]

    return backSubst(M)

def gaussPartialPivot(Ma, b):
    #Getting matrix dimention
    n = len(Ma)

    #Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))

    # Matrix reduction
    for i in range(n-1):
        # Row swaping
        maxV = float('-inf')    # Max value in the column
        maxI = None             # Index of the max value
        for j in range(i+1,n):
            if(maxV<abs(M[j,i])):
                maxV = M[j,i]
                maxI = j
        if (maxV>abs(M[i,i])):
            aux = np.copy(M[maxI,i:n+1])
            M[maxI,i:n+1] = M[i,i:n+1]
            M[i,i:n+1] = aux

        for j in range(i+1,n):
            if (M[j,i] != 0):
                M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]
    # Back substitut
    return backSubst(M)

def gaussTotalPivot(Ma, b):
    #Getting matrix dimention
    n = len(Ma)

    #Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))
    changes = np.array([])
    
    # Matrix reduction
    for i in range(n-1):
        # Column swaping
        maxV = float('-inf')    # Max value in the in the sub matrix
        maxI = None             # Index of the max value

        for j in range(i+1,n):
            for k in range(i, n):
                if(maxV<abs(M[k,j])):
                    maxV = M[k,j]
                    maxI = (k,j)
        if (maxV>abs(M[i,i])):
            a, b = maxI
            changes = np.vstack((changes, np.array([i,b]))) if len(changes) else np.array([i,b])
            aux = np.copy(M[:,b])
            M[:,b] = M[:,i]
            M[:,i] = aux

        # Row swaping
        maxV = float('-inf')    # Max value in the in the same column
        maxI = None             # Index of the max value
        for l in range(i+1,n):
            if(maxV<abs(M[l,i])):
                maxV = M[l,i]
                maxI = l
        if (maxV>abs(M[i,i])):
            aux = np.copy(M[maxI,i:n+1])
            M[maxI,i:n+1] = M[i,i:n+1]
            M[i,i:n+1] = aux

        for j in range(i+1,n):
            if (M[j,i] != 0):
                M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]
    
    print(changes)
    # Back substitution
    x = backSubst(M)

    # Reorganize the solution
    for i in changes[::-1]:
        aux = np.copy(x[i[0]])
        x[i[0]] = x[i[1]]
        x[i[1]] = aux
    
    return x

def backSubst(M):
    # Getting  matrix dimention
    n = len(M)
    # Initializing a zero vector
    x = np.matlib.zeros((n,1))
    x[n-1]=M[n-1,n]/M[n-1,n-1]
    for i in range(n-2, -1, -1):
        aux1 = np.hstack((1, np.asarray(x[i+1:n]).reshape(-1)))
        aux2 = np.hstack((M[i,n], np.asarray(-M[i,i+1:n]).reshape(-1)))
        x[i] = np.dot(aux1,aux2)/M[i,i]
    return x

matrix = [[2,-1,0,3],
         [1,0.5,3,8],
         [0,13,-2,11],
         [14,5,-2,3]]

vector = [1,1,1,1]

print(gaussTotalPivot(matrix,vector))