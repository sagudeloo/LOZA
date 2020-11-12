import numpy as np
import numpy.matlib
from scipy import linalg

def Lagrange(X,Y):
    output = {
        "type": 5,
        "method": "Lagrange"
    }
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
    output ["results"] = L
    output ["x"] = Y
    Coef = Y.dot(L)
    return output

def Trazlin(X,Y):
    output = {
        "type": 5,
        "method": "Tracers"
    }
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

    output["results"] = Coef
    return output

def TrazlinQuadratic(X,Y):
    output = {
        "type": 6,
        "method": "Tracers"
    }
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

    output["results"] = Coef
    return output

def TrazlinCubicos(X,Y):
    output = {
        "type": 5,
        "method": "Tracers"
    }
    n = X.size
    m = 4*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,4))
    i = 0
    #Condicion de interpolacion
    while i < X.size-1:
        
        A[i+1,4*i:4*i+4]= np.hstack((X[i+1]**3,X[i+1]**2,X[i+1],1)) 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,0:4] = np.hstack((X[0]**3,X[0]**2,X[0]**1,1))
    b[0] = Y[0]
    #Condicion de continuidad
    i = 1
    while i < X.size-1:
        A[X.size-1+i,4*i-4:4*i+4] = np.hstack((X[i]**3,X[i]**2,X[i],1,-X[i]**3,-X[i]**2,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1
    #Condicion de suavidad
    i = 1
    while i < X.size-1:
        A[2*n-3+i,4*i-4:4*i+4] = np.hstack((3*X[i]**2,2*X[i],1,0,-3*X[i]**2,-2*X[i],-1,0))
        b[2*n-3+i] = 0
        i = i+1
    
    #Condicion de concavidad
    i = 1
    while i < X.size-1:
        A[3*n-5+i,4*i-4:4*i+4] = np.hstack((6*X[i],2,0,0,-6*X[i],-2,0,0))
        b[n+5+i] = 0
        i = i+1
    
    #Condiciones de frontera   
    A[m-2,0:2]=[6*X[0],2];
    b[m-2]=0;
    A[m-1,m-4:m-2]=[6*X[X.size-1],2];
    b[m-1]=0;
    
    Saux = linalg.solve(A,b)
    #Mostrar Coeficientes
    i = 0
    j = 0
    while i < n-1:
        Coef[i,:] = np.hstack((Saux[j],Saux[j+1],Saux[j+2],Saux[j+3]))
        i = i+1
        j = j + 4

    output["results"] = Coef
    return output

def outputLarange(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput += "\nResults:\n"
    stringOutput += "\nLagrange interpolating polynomials:\n\n"
    rel = output["results"]
    i = 0
    while i < len(rel) :
        stringOutput += '{:^6f}'.format(rel[i,0]) + "x^3"
        stringOutput += format(rel[i,1],"+.6f") + "x^2"
        stringOutput += format(rel[i,2],"+.6f") + "x"
        stringOutput += format(rel[i,3],"+.6f") +"   //L" + str(i) + "\n"
        i += 1

    stringOutput += "\n Polynomial:\n"
    j = 0
    for i in output["x"]:
        stringOutput += str(i) +'*L+' + str(j)
        j += 1
    stringOutput += "\n______________________________________________________________\n"
    return stringOutput

def outputTracers(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput += "\nResults:\n"
    stringOutput += "\nTracer coefficients:\n\n"
    rel = output["results"]
    i = 0
    aux = rel.shape
    while i < aux[0] :
        j = 0
        while j < aux[1]:
            stringOutput += '{:^6f}'.format(rel[i,j]) +"  "
            j += 1
        i += 1
        stringOutput += "\n"
    stringOutput += "\n Tracers:\n"
    i = 0
    while i < aux[0] :
        j = 0
        if aux[1] == 2:
            stringOutput += format(rel[i,0],"6f") +"x"
            stringOutput += format(rel[i,1],"+.6f") 
        elif aux[1] == 3:
            stringOutput += format(rel[i,0],"6f") +"x^2"
            stringOutput += format(rel[i,1],"+.6f") +"x"
            stringOutput += format(rel[i,2],"+.6f")
        elif aux[1] == 4:
            stringOutput += format(rel[i,0],"6f") +"x^3"
            stringOutput += format(rel[i,1],"+.6f") +"x^2"
            stringOutput += format(rel[i,2],"+.6f") +"x"
            stringOutput += format(rel[i,3],"+.6f")

        i += 1
        stringOutput += "\n"
    stringOutput += "\n______________________________________________________________\n"

    return stringOutput


X = np.array([-1, 0, 3, 4])
Y = np.array([15.5, 3, 8, 1])
output = Trazlin(X,Y)
print(outputTracers(output))
