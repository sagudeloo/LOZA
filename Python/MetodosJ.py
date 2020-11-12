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

def Trazlin(X,Y):
    n = X.size
    m = 2*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,2))
    i = 0
    #Condicion de interpolacion
    while i < X.size-1:
        A[i+1,[2*i+1-1,2*i+1]]= [X[i+1],1] 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,[0,1]] = [X[0],1] 
    b[0] = Y[0]
    i = 1
    #Condicion de continuidad
    while i < X.size-1:
        A[X.size-1+i,2*i-2:2*i+2] = np.hstack((X[i],1,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1

    Saux = linalg.solve(A,b)
    #Mostrar Coeficientes
    i = 0
    while i < X.size-1:
        Coef[i,:] = [Saux[2*i],Saux[2*i+1]]
        i = i+1
    print(Coef)

    #Mostrar trazadores
    i = 0
    while i < X.size-1:
        print(Coef[i,0],"x+",Coef[i,1])
        i = i+1

def TrazlinCuadratico(X,Y):
    n = X.size
    m = 3*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,3))
    i = 0
    #Condicion de interpolacion
    while i < X.size-1:
        
        A[i+1,3*i:3*i+3]= np.hstack((X[i+1]**2,X[i+1],1)) 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,0:3] = np.hstack((X[0]**2,X[0]**1,1))
    b[0] = Y[0]
    #Condicion de Continuidad
    i = 1
    while i < X.size-1:
        A[X.size-1+i,3*i-3:3*i+3] = np.hstack((X[i]**2,X[i],1,-X[i]**2,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1
    #Condicion de suavidad
    i = 1
    while i < X.size-1:
        A[2*n-3+i,3*i-3:3*i+3] = np.hstack((2*X[i],1,0,-2*X[i],-1,0))
        b[2*n-3+i] = 0
        i = i+1
    A[m-1,0]=2;
    b[m-1]=0;

    Saux = linalg.solve(A,b)
    #Mostrar Coeficientes
    i = 0
    j = 0
    while i < n-1:
        Coef[i,:] = np.hstack((Saux[j],Saux[j+1],Saux[j+2]))
        i = i+1
        j = j + 3
    print("Coeficientes de los trazadores: ")
    print(Coef)

    #Mostrar trazadores
    i = 0
    print("Trazadores: ")
    while i < X.size-1:
        print(Coef[i,0],"x^2+",Coef[i,1],"x+",Coef[i,2])
        i = i+1


X = np.array([-1, 0, 3, 4])
Y = np.array([15.5, 3, 8, 1])
#Lagrange(X,Y)
#Trazlin(X,Y)
TrazlinCuadratico(X,Y)