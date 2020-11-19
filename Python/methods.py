import numpy as np
import numpy.matlib
from numpy.lib import scimath
import cmath
from sympy import *

def mulRoots(fx, x0, numMax):

    output = {
        "type": 1,
        "method": "Multi Roots",
        "columns": ["iter","xi","f(xi)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 0
    cond = 0.0000001
    error = 1.0000000
    ex = sympify(fx)

    d_ex = diff(ex, x)
    d2_ex = diff(d_ex, x)

    y = x0
    ex_2 = 0
    d_ex2 = 0
    d2_ex2 = 0

    d_ex = diff(ex, x)
    d2_ex = diff(d_ex, x)


    while error > cond and i < numMax:
        if i == 0:
            ex_2 = ex.subs(x, x0)
            ex_2 = ex_2.evalf()
            results.append([i, x0, ex_2])
        else:
            d_ex2 = d_ex.subs(x, x0)
            d_ex2 = d_ex2.evalf()

            d2_ex2 = d2_ex.subs(x, x0)
            d2_ex2 = d2_ex2.evalf()

            y2 = sympify(y)
            y = y2 - ((ex_2 * d_ex2) / Pow(d_ex2, 2) - (ex_2*d2_ex2))

            ex_2 = ex_2.subs(x0, y)
            error = Abs(y - x0)
            ex_2 = ex_2.evalf()
            er = sympify(error)
            er.evalf()
            error = er
            ex = ex_2
            x0 = y
            results.append(i , y, ex_2, error)

        i += 1
    output["results"] = results
    output["root"] = y
    return output

def newton(fx, x0, numMax):

    output = {
        "type": 1,
        "method": "Newton",
        "columns": ["iter","xi","f(xi)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 0
    cond = 0.0000001
    error = 1.0000000

    ex = sympify(fx)

    d_ex = diff(ex, x)

    y = x0
    ex_2 = ex
    d_ex2 = ex

    while((error > cond) and (i < numMax)):
        if i == 0:
            ex_2 = ex.subs(x, x0)
            ex_2 = ex_2.evalf()
            d_ex2 = d_ex.subs(x, x0)
            d_ex2 = d_ex2.evalf()
            results.append([i, x0, ex_2])
        else:
            y2 = sympify(y)
            y = y2 - (ex_2/d_ex2)

            ex_2 = ex.subs(x0, y)
            ex_2 = ex.evalf()
            d_ex2 = d_ex2.subs(x0, y)

            d_ex2 = d_ex2.evalf(x0, y)

            error = Abs(y - x0)
            er = sympify(error)
            error = er.evalf()
            print(str(error))
            ex = ex_2
            d_ex = d_ex2
            x0 = y
            results.append(i , y, ex_2, error)



        i += 1
    output["results"] = results
    output["root"] = y
    return output

def fixedPoint(fx, gx, x0, numMax):

    output = {
        "type": 1,
        "method": "Fixed point",
        "columns": ["iter","xi","f(xi)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 0
    cond = 0.0000001
    error = 1.0000000
    ex = sympify(fx)

    y = x0
    ex_2 = 0

    rx = sympify(gx)

    y = x0
    ex_2 = 0

    rx = sympify(gx)

    rx_2 = 0

    while((error > cond) and (i < numMax)):
        if i == 0:

            ex_2 = ex.subs(x, y)
            ex_2 = ex_2.evalf()

            rx_2 = rx.subs(x, y)
            rx_2 = rx_2.evalf()

            results.append([i, x0, ex_2])
        else:
            y = rx_2.evalf()

            ex_2 = ex.subs(x, y)
            ex_2 = ex_2.evalf()

            rx_2 = rx.subs(x, y)
            rx_2 = rx_2.evalf()
            error = Abs(y - x0)

            er = sympify(error)
            error = er.evalf()

            x0 = y

            results.append([i, x0, ex_2])
        i += 1

    output["resultado"] = results
    output["root"] = y
    return output

def incremSearch(fx, x0, d, numMax):

    output = {
        "type": 0,
        "method": "Incremental Search",
    }
    results = list()
    x = Symbol('x')
    i = 0

    ex = sympify(fx)

    y = x0
    ex_2 = 0.1
    ex_3 = 0.1
    while (i < numMax):
        if i == 0:
            ex_2 = ex.subs(x, y)
            ex_2 = ex_2.evalf()
        else:
            x0 = y
            y = y + d

            ex_3 = ex_2

            ex_2 = ex.subs(x, y)
            ex_2 = ex_2.evalf()

            if (ex_2*ex_3 < 0):
                results.append([x0, y])

        i += 1
    output["results"] = results
    return output

def bisec(a, b, fx, numMax):

    output = {
        "type": 1,
        "method": "Bisection",
        "columns": ["iter","a","xm", "b","f(xm)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 1
    cond = 0.0000001
    error = 1.0000000

    ex = sympify(fx)

    xm = 0
    xm0 = 0
    ex_2 = 0
    ex_3 = 0

    w = 0

    while (error > cond) and (i < numMax):
        if i == 0:
            xm = (a + b)/2

            ex_2 = ex.subs(x, xm)
            ex_3 = ex_2.evalf()

            ex_2 = ex.subs(x, a)
            ex_2 = ex_2.evalf()
            results.append([i, a, xm, b, ex_3])
        else:

            if (w < 0):
                b = xm
            else:
                a = xm

            xm0 = xm
            xm = (a+b)/2

            ex_2 = ex.subs(x, xm)
            ex_3 = ex_2.evalf()

            ex_2 = ex.subs(x, a)
            ex_2 = ex_2.evalf()

            error = Abs(xm-xm0)
            er = sympify(error)
            error = er.evalf()

            if (ex_2*ex_3 < 0):
                w = -1
            else:
                w = 1
            results.append([i, a, xm, b, ex_3, error])
        i += 1

    output["results"] = results
    output["root"] = xm
    return output

def regulaFalsi(a, b, fx, numMax):

    output = {
        "type": 1,
        "method": "Regula falsi",
        "columns": ["iter","a","xm", "b","f(xm)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 1
    cond = 0.0000001
    error = 1.0000000

    ex = sympify(fx)

    xm = 0
    xm0 = 0
    ex_2 = 0
    ex_3 = 0
    ex_a = 0
    ex_b = 0

    while (error > cond) and (i < numMax):
        if i == 1:
            ex_2 = ex.subs(x, a)
            ex_2 = ex_2.evalf()
            ex_a = ex_2

            ex_2 = ex.subs(x, b)
            ex_2 = ex_2.evalf()
            ex_b = ex_2

            xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)
            ex_3 = ex.subs(x, xm)
            ex_3 = ex_3.evalf()
            results.append([i, a, xm, b, ex_3])
        else:

            if (ex_a*ex_3 < 0):
                b = xm
            else:
                a = xm

            xm0 = xm
            ex_2 = ex.subs(x, a)
            ex_2 = ex_2.evalf()
            ex_a = ex_2

            ex_2 = ex.subs(x, b)
            ex_2 = ex_2.evalf()
            ex_b = ex_2

            xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)

            ex_3 = ex.subs(x, xm)
            ex_3 = ex_3.evalf()

            error = Abs(xm-xm0)
            er = sympify(error)
            error = er.evalf()
            results.append([i, a, xm, b, ex_3, error])
        i += 1

    output["results"] = results
    output["root"] = xm
    return output

def secan(x0, x1, fx, numMax):

    output = {
        "type": 1,
        "method": "Secant",
        "columns": ["iter", "xi", "f(xi)","E" ],
        "iterations": numMax
    }
    results = list()
    x = Symbol('x')
    i = 0
    cond = 0.0000001
    error = 1.0000000

    ex = sympify(fx)

    y = x0
    ex_0 = ex
    ex_1 = ex

    while((error > cond) and (i < numMax)):
        if i == 0:
            ex_0 = ex.subs(x, x0)
            ex_0 = ex_0.evalf()
            results.append([i, x0, ex_0])
        elif i == 1:
            ex_1 = ex.subs(x, x1)
            ex_1 = ex_1.evalf()
            results.append([i, x1, ex_1])
        else:
            y = x1
            x1 = x1 - (ex_1*(x1 - x0)/(ex_1 - ex_0))
            x0 = y

            ex_0 = ex_1.subs(x, x0)
            ex_0 = ex_1.evalf()

            ex_1 = ex.subs(x, x1)
            ex_1 = ex_1.evalf()

            error = Abs(x1 - x0)
            er = sympify(error)
            error = er.evalf()
            results.append([i, x1, ex_1, error])
        i += 1

    output["results"] = results
    output["root"] = y
    return output


def gaussSimple(Ma, b):

    output = {
        "type": 2,
        "method": "Simple Gaussian Reduction"
    }

    # Getting matrix dimention
    n = len(Ma)

    # Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa, vectorB))

    steps = {'Step 0': np.copy(M)}

    # Matrix reduction
    for i in range(n-1):
        for j in range(i+1, n):
            if (M[j, i] != 0):
                M[j, i:n+1] = M[j, i:n+1]-(M[j, i]/M[i, i])*M[i, i:n+1]
        steps[f'Step {i+1}'] = np.copy(M)

    output["results"] = steps

    output["x"] = backSubst(M)

    return output


def gaussPartialPivot(Ma, b):

    output = {
        "type": 2,
        "method": "Gaussian Reduction With Partial Pivoting"
    }

    # Getting matrix dimention
    n = len(Ma)

    # Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa, vectorB))

    steps = {'Step 0': np.copy(M)}

    # Matrix reduction
    for i in range(n-1):
        # Row swapping
        maxV = float('-inf')    # Max value in the column
        maxI = None             # Index of the max value
        for j in range(i+1, n):
            if(maxV < abs(M[j, i])):
                maxV = M[j, i]
                maxI = j
        if (maxV > abs(M[i, i])):
            aux = np.copy(M[maxI, i:n+1])
            M[maxI, i:n+1] = M[i, i:n+1]
            M[i, i:n+1] = aux

        for j in range(i+1, n):
            if (M[j, i] != 0):
                M[j, i:n+1] = M[j, i:n+1]-(M[j, i]/M[i, i])*M[i, i:n+1]
        steps[f'Step {i+1}'] = np.copy(M)

    output["results"] = steps

    output["x"] = backSubst(M)

    return output


def gaussTotalPivot(Ma, b):

    output = {
        "type": 2,
        "method": "Gaussian Reduction With Total Pivoting"
    }

    # Getting matrix dimention
    n = len(Ma)

    # Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa, vectorB))
    changes = np.array([])

    steps = {'Step 0': np.copy(M)}

    # Matrix reduction
    for i in range(n-1):
        # Column swaping
        maxV = float('-inf')    # Max value in the in the sub matrix
        maxI = None             # Index of the max value

        for j in range(i+1, n):
            for k in range(i, n):
                if(maxV < abs(M[k, j])):
                    maxV = M[k, j]
                    maxI = (k, j)
        if (maxV > abs(M[i, i])):
            a, b = maxI
            changes = np.vstack((changes, np.array([i, b]))) if len(
                changes) else np.array([i, b])
            aux = np.copy(M[:, b])
            M[:, b] = M[:, i]
            M[:, i] = aux

        # Row swaping
        maxV = float('-inf')    # Max value in the in the same column
        maxI = None             # Index of the max value
        for l in range(i+1, n):
            if(maxV < abs(M[l, i])):
                maxV = M[l, i]
                maxI = l
        if (maxV > abs(M[i, i])):
            aux = np.copy(M[maxI, i:n+1])
            M[maxI, i:n+1] = M[i, i:n+1]
            M[i, i:n+1] = aux

        for j in range(i+1, n):
            if (M[j, i] != 0):
                M[j, i:n+1] = M[j, i:n+1]-(M[j, i]/M[i, i])*M[i, i:n+1]
        steps[f'Step {i+1}'] = np.copy(M)

    output["results"] = steps

    # Back substitution
    x = backSubst(M)

    # Reorganize the solution
    for i in changes[::-1]:
        aux = np.copy(x[i[0]])
        x[i[0]] = x[i[1]]
        x[i[1]] = aux

    output["x"] = x

    return output

def jacobiM(Ma, Vb, x0, tol, numMax):
    sX = x0.size
    xA = np.zeros((sX,1))
    
    A = np.matrix(Ma)
    
    b = np.array(Vb)
    s = b.size
    b = np.reshape(b,(s,1))

    D = np.diag(np.diag(A))
    L = -1*np.tril(A)+D
    U = -1*np.triu(A)+D
    LU = L+U
    
    T = np.linalg.inv(D) @ LU
    C = np.linalg.inv(D) @ b
    
    xP = x0
    E = 1000
    cont = 0

    while(E > tol and cont < numMax  ):
         xA = T@xP + C
         E = np.linalg.norm(xP - xA)
         xP = xA
         cont = cont + 1

    x = xA
    nIter = cont
    error = E
    
    print(f"x =  {x}  \n\nIteración n = {nIter} \nError = {error}  \n\nT = {T} \n\n C = {C}")

def gaussSei(Ma, Vb, x0, tol, numMax):
    sX = x0.size
    xA = np.zeros((sX,1))
    
    A = np.matrix(Ma)
    
    b = np.array(Vb)
    s = b.size
    b = np.reshape(b,(s,1))

    D = np.diag(np.diag(A))
    L = -1*np.tril(A)+D
    U = -1*np.triu(A)+D
    
    T = np.linalg.inv(D-L) @ U
    C = np.linalg.inv(D-L) @ b
    
    xP = x0
    E = 1000
    cont = 0

    while(E > tol and cont < numMax  ):
         xA = T@xP + C
         E = np.linalg.norm(xP - xA)
         xP = xA
         cont = cont + 1

    x = xA
    nIter = cont
    error = E
    
    print(f"x =  {x}  \n\nIteración n = {nIter} \nError = {error}  \n\nT = {T} \n\n C = {C}")

def sorM(Ma, Vb, x0, w, tol, numMax):
    sX = x0.size
    xA = np.zeros((sX,1))
    
    A = np.matrix(Ma)
    
    b = np.array(Vb)
    s = b.size
    b = np.reshape(b,(s,1))

    D = np.diag(np.diag(A))
    L = -1*np.tril(A)+D
    U = -1*np.triu(A)+D
    
    T = np.linalg.inv(D-(w*L)) @ (((1-w)*D)+(w*U))
    C = (w*np.linalg.inv(D-(w*L))) @ b
    
    xP = x0
    E = 1000
    cont = 0

    while(E > tol and cont < numMax  ):
         xA = T@xP + C
         E = np.linalg.norm(xP - xA)
         xP = xA
         cont = cont + 1

    x = xA
    nIter = cont
    error = E
    
    print(f"x =  {x}  \n\nIteración n = {nIter} \nError = {error}  \n\nT = {T} \n\n C = {C}")

def vanderMon(Vx,Vy):
    X = np.array(Vx)
    s1 = X.size

    Y = np.array(Vy)

    A = np.zeros((s1,s1))

    i = 0
    for i in range(i,s1):
        A[:,i]=X**(s1-i-1)
    
    Coef = (np.linalg.solve(A,Y)).conj().transpose()
    
    print(f"\nA :{A} \nCoef :{Coef}")

def difdividas(Vx,Vy):
    X = np.array(Vx)
    n = X.size

    Y = np.array(Vy)

    D = np.zeros((n,n))

    D[:,0]=Y.T
    for i in range(1,n):
        aux0 = D[i-1:n,i-1]
        aux = np.diff(aux0)
        aux2 = X[i:n] - X[0:n-i]
        D[i:n,i] = aux/aux2.T

    Coef = np.diag(D)
    
    
    print(f"\nD :{D} \nCoef :{Coef}")

def backSubst(M):
    n = M.shape[0]
    # Getting  matrix dimention
    # Initializing a zero vector
    x = np.matlib.zeros((n, 1), dtype=complex)
    x[n-1] = M[n-1, n]/M[n-1, n-1]
    for i in range(n-2, -1, -1):
        aux1 = np.hstack((1, np.asarray(x[i+1:n]).reshape(-1)))
        aux2 = np.hstack((M[i, n], np.asarray(-M[i, i+1:n]).reshape(-1)))
        x[i] = np.dot(aux1, aux2)/M[i, i]
    return x

def forSubst(M):
    # Getting  matrix dimention
    n = M.shape[0]
    # Initializing a zero vector
    x = np.matlib.zeros((n, 1), dtype=complex)
    x[0] = M[0, n]/M[0, 0]
    for i in range(1, n, 1):
        aux1 = np.hstack((1, np.asarray(x[0:i]).reshape(-1)))
        aux2 = np.hstack((M[i, n], np.asarray(-M[i, 0:i]).reshape(-1)))
        x[i] = np.dot(aux1, aux2)/M[i, i]
    return x

def LUSimple(Ma, b):

    output = {
        "type": 3,
        "method": "LU With Gaussian Simple"
    }

    # Initialization
    matrixMa = np.array(Ma)
    vectorB = np.array(b).T
    n = matrixMa.shape[0]
    L = np.eye(n)
    U = np.zeros((n,n))
    M = matrixMa
    steps = {'Step 0': [np.copy(M)]}
    
    # Factorization
    for i in range(n-1):
        for j in range(i+1, n):
            if not (M[j,i] == 0):
                L[j,i]=M[j,i]/M[i,i]
                M[j,i:n]=M[j,i:n]-(M[j,i]/M[i,i])*M[i,i:n]
        U[i, i:n]=M[i,i:n]
        U[i+1,i+1:n]=M[i+1,i+1:n]
        steps[f"Step {i+1}"] = [np.copy(M),{"L:":np.copy(L)},{"U:":np.copy(U)}]
    U[n-1,n-1]=M[n-1,n-1]
    
    output["results"] = steps

    # Resoults delivery
    z=forSubst(np.column_stack((L,b)))
    x=backSubst(np.column_stack((U,z)))

    output["x"] = x

    return output

def LUPartialPivot(Ma, b):

    output = {
        "type": 3,
        "method": "LU With Partial Pivot"
    }

    # Initialization
    matrixMa = np.array(Ma)
    vectorB = np.array(b).T
    n = matrixMa.shape[0]
    L = np.eye(n)
    U = np.zeros((n,n))
    P = np.eye(n)
    M = matrixMa

    steps = {'Step 0': [np.copy(M)]}

    # Factorization
    for i in range(0,n-1):
        # row swapping
        maxV = float('-inf')    # Max value in the column
        maxI = None             # Index of the max value
        for j in range(i+1, n):
            if(maxV < abs(M[j, i])):
                maxV = M[j, i]
                maxI = j
        if (maxV > abs(M[i, i])):
            aux2=np.copy(M[maxI,i:n])
            aux3=np.copy(P[maxI,:])
            M[maxI,i:n]=M[i,i:n]
            P[maxI,:]=P[i,:]
            M[i,i:n]=aux2
            P[i,:]=aux3
            if i>0:
                aux4=L[maxI, 0:i-1]
                L[maxI, 0:i-1]=L[i,0:i-1]
                L[i,0:i-1]=aux4
        for j in range(i+1, n):
            if not (M[j,i] == 0):
                L[j,i]=M[j,i]/M[i,i]
                M[j,i:n]=M[j,i:n]-(M[j,i]/M[i,i])*M[i,i:n]
        U[i, i:n]=M[i,i:n]
        U[i+1,i+1:n]=M[i+1,i+1:n]
        steps[f"Step {i+1}"] = [np.copy(M),{"L:":np.copy(L)},{"U:":np.copy(U)},{"P:":np.copy(P)}]

    U[n-1,n-1]=M[n-1,n-1]

    output["results"] = steps
    
    # Resoults delivery
    z=forSubst(np.column_stack((L,P@vectorB)))
    x=backSubst(np.column_stack((U,z)))

    output["x"] = x

    return output

def crout(Ma, b):

    output = {
        "type": 3,
        "method": "Crout"
    }

    # Initialization
    A = np.array(Ma)
    n = A.shape[0]
    L = np.eye(n)
    U = np.eye(n)

    steps = {'Step 0': [np.copy(A)]}

    # Factorization
    for i in range(n-1):
        for j in range(i, n):
            L[j,i]=A[j,i]-np.dot(L[j,0:i], U[0:i,i].T);
        for j in range(i+1, n):
            U[i,j]=(A[i,j]-np.dot(L[i,0:i], U[0:i,j].T))/L[i,i]
        steps[f"Step {i+1}"] = [{"L:":np.copy(L)},{"U:":np.copy(U)}]
    L[n-1,n-1]=A[n-1,n-1]-np.dot(L[n-1,0:n-1], U[0:n-1,n-1].T)

    output["results"] = steps

    z=forSubst(np.column_stack((L,b)))
    x=backSubst(np.column_stack((U, z)))

    output["x"] = x

    return output

def doolittle(Ma, b):

    output = {
        "type": 3,
        "method": "Doolittle"
    }

    # Initialization
    A = np.array(Ma)
    n = A.shape[0]
    L = np.eye(n)
    U = np.eye(n)

    steps = {'Step 0': [np.copy(A)]}

    # Factorization
    for i in range(n-1):
        for j in range(i, n):
            U[i,j]=A[i,j]-np.dot(L[i,0:i], U[0:i,j].T)
        for j in range(i+1, n):
            L[j,i]=(A[j,i]-np.dot(L[j,0:i], U[0:i,i].T))/U[i,i]
        steps[f"Step {i+1}"] = [{"L:":np.copy(L)},{"U:":np.copy(U)}]
        
    U[n-1,n-1]=A[n-1,n-1]-np.dot(L[n-1,0:n-1], U[0:n-1,n-1].T)

    output["results"] = steps

    z=forSubst(np.column_stack((L,b)))
    x=backSubst(np.column_stack((U, z)))

    output["x"] = x

    return output

def cholesky(Ma, b):

    output = {
        "type": 3,
        "method": "Cholesky"
    }

    # Initialization
    A = np.array(Ma)
    n = A.shape[0]
    L = np.eye(n, dtype=complex)
    U = np.eye(n, dtype=complex)

    steps = {'Step 0': [np.copy(A)]}

    # Factorization
    for i in range(n-1):
        L[i,i]= scimath.sqrt(A[i,i]-np.dot(L[i,0:i], U[0:i,i].T))
        U[i,i]=L[i,i]
        for j in range(i+1, n):
            L[j,i]=(A[j,i]-np.dot(L[j,0:i], U[0:i,i].T))/U[i,i];
        for j in range(i+1, n):
            U[i,j]=(A[i,j]-np.dot(L[i,0:i], U[0:i,j].T))/L[i,i]
        steps[f"Step {i+1}"] = [{"L:":np.copy(L)},{"U:":np.copy(U)}]
    L[n-1,n-1]=scimath.sqrt(A[n-1,n-1]-np.dot(L[n-1,0:n-1], U[0:n-1,n-1].T))
    U[n-1,n-1]=L[n-1,n-1]

    output["results"] = steps

    z=forSubst(np.column_stack((L,b)))
    x=backSubst(np.column_stack((U, z)))

    output["x"] = x

    return output

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
        "type": 6,
        "method": "Tracers"
    }
    n = X.size
    m = 2*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,2))
    i = 0
    #Interpolating condition
    while i < X.size-1:
        A[i+1,[2*i+1-1,2*i+1]]= [X[i+1],1] 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,[0,1]] = [X[0],1] 
    b[0] = Y[0]
    i = 1
    #Condition of continuity
    while i < X.size-1:
        A[X.size-1+i,2*i-2:2*i+2] = np.hstack((X[i],1,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1

    Saux = linalg.solve(A,b)
    #Order Coefficients
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
    #Interpolating condition
    while i < X.size-1:
        
        A[i+1,3*i:3*i+3]= np.hstack((X[i+1]**2,X[i+1],1)) 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,0:3] = np.hstack((X[0]**2,X[0]**1,1))
    b[0] = Y[0]
    #Condition of continuity
    i = 1
    while i < X.size-1:
        A[X.size-1+i,3*i-3:3*i+3] = np.hstack((X[i]**2,X[i],1,-X[i]**2,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1
    #Condition of smoothness
    i = 1
    while i < X.size-1:
        A[2*n-3+i,3*i-3:3*i+3] = np.hstack((2*X[i],1,0,-2*X[i],-1,0))
        b[2*n-3+i] = 0
        i = i+1
    A[m-1,0]=2;
    b[m-1]=0;

    Saux = linalg.solve(A,b)
    #Order Coefficients
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
        "type": 6,
        "method": "Tracers"
    }
    n = X.size
    m = 4*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,4))
    i = 0
    #Interpolating condition
    while i < X.size-1:
        
        A[i+1,4*i:4*i+4]= np.hstack((X[i+1]**3,X[i+1]**2,X[i+1],1)) 
        b[i+1]=Y[i+1]
        i = i+1

    A[0,0:4] = np.hstack((X[0]**3,X[0]**2,X[0]**1,1))
    b[0] = Y[0]
    #Condition of continuity
    i = 1
    while i < X.size-1:
        A[X.size-1+i,4*i-4:4*i+4] = np.hstack((X[i]**3,X[i]**2,X[i],1,-X[i]**3,-X[i]**2,-X[i],-1))
        b[X.size-1+i] = 0
        i = i+1
    #Condition of smoothness
    i = 1
    while i < X.size-1:
        A[2*n-3+i,4*i-4:4*i+4] = np.hstack((3*X[i]**2,2*X[i],1,0,-3*X[i]**2,-2*X[i],-1,0))
        b[2*n-3+i] = 0
        i = i+1
    
    #Concavity condition
    i = 1
    while i < X.size-1:
        A[3*n-5+i,4*i-4:4*i+4] = np.hstack((6*X[i],2,0,0,-6*X[i],-2,0,0))
        b[n+5+i] = 0
        i = i+1
    
    #Boundary conditions  
    A[m-2,0:2]=[6*X[0],2];
    b[m-2]=0;
    A[m-1,m-4:m-2]=[6*X[X.size-1],2];
    b[m-1]=0;
    
    Saux = linalg.solve(A,b)
    #Order Coefficients
    i = 0
    j = 0
    while i < n-1:
        Coef[i,:] = np.hstack((Saux[j],Saux[j+1],Saux[j+2],Saux[j+3]))
        i = i+1
        j = j + 4

    output["results"] = Coef
    return output


def outputToString(output):
    type = output["type"]
    if(type==0):
        return outputIncrementalSearch(output)
    elif (type==1):
        return outputAnalyticalMethod(output)
    elif (type==2):
        return outputGauss(output)
    elif (type==3):
        return outputLU(output)
    elif (type==4):
        pass
    elif (type==5):
        return outputLarange(output)
    elif (type==6):
        return outputTracers(output)


def outputIncrementalSearch(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput = "\nResults:\n"
    results = output["results"]
    for i in results:
        stringOutput += f'There is a root of f in {i}\n'
    stringOutput += "\n______________________________________________________________\n"
    return stringOutput


def outputAnalyticalMethod(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput = "\nResults table:\n"
    columns = output["columns"]
    for i in columns:
        stringOutput += '|{:^25}'.format(i)
    stringOutput += "|\n"
    results = output["results"]
    for j in results:
        for k in j:
            stringOutput += '|{:^25E}'.format(k)
        stringOutput += "|\n"
    if ("root" in output):
        stringOutput += f'\nAn approximation of the root was fund in: {output["root"]}\n'
    else:
        stringOutput += f'\nAn approximation of the root was not found in {output["iterations"]}\n'
    stringOutput += "\n______________________________________________________________\n"
    return stringOutput


def outputGauss(output):
    stringOutput = f'\n{output["method"]}'
    stringOutput += "\nResults:\n"
    results = output["results"]
    for i, j in results.items():
        stringOutput += f'\n{i}\n\n'
        for k in j:
            for m in k:
                stringOutput += '{:^25E}'.format(m.item())
            stringOutput += "\n"
    stringOutput += "\nAfter Backward Sustitution\n"
    stringOutput += "\nx:\n"
    for i in output["x"]:
        stringOutput += '{:^25E}\n'.format(i.item())

    stringOutput += "\n______________________________________________________________\n"

    return stringOutput

def outputLU(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput += "\nResults:\n"
    results = output["results"]
    for i,j in results.items():
        stringOutput += f'\n{i}\n\n'
        for k in j:
            if(isinstance(k,numpy.ndarray)):
                for l in k:
                    for m in l:
                        real = f'{(m.item()).real:.6f}' if not (m.item()).real == 0 else ""
                        imag = f'{(m.item()).imag:+.6f}j' if not (m.item()).imag == 0 else ""
                        num = real+imag if real or imag else f'{0:.6f}'
                        stringOutput+= f'{num:^15}'
                    stringOutput += "\n"
            else:
                for l,m in k.items():
                    stringOutput += f'\n{l}\n'
                    for n in m:
                        for o in n:
                            real = f'{(o.item()).real:.6f}' if not (o.item()).real == 0 else ""
                            imag = f'{(o.item()).imag:+.6f}j' if not (o.item()).imag == 0 else ""
                            num = real+imag if real or imag else f'{0:.6f}'
                            stringOutput+= f'{num:^15}'
                        stringOutput += "\n"
    stringOutput += "\nAfter Backward And Forward Sustitution\n"
    stringOutput += "\nx:\n"
    for i in output["x"]:
        real = f'{(i.item()).real:.6f}' if not (i.item()).real == 0 else ""
        imag = f'{(i.item()).imag:+.6f}j' if not (i.item()).imag == 0 else ""
        num = real+imag if real or imag else f'{0:.6f}'
        stringOutput += f'{num:^15}\n'

    stringOutput += "\n______________________________________________________________\n"

    return stringOutput

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
