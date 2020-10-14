from sympy import *
import numpy as np
import numpy.matlib

def mulRoots(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 d_ex = diff(ex, x)
 d2_ex = diff(d_ex, x)
 
 y = x0
 ex_2 = ex
 d_ex2 = ex
 d2_ex2 = ex 

 while error > cond and i < numMax:
     if i == 0:
         ex_2 = ex.subs(x,x0)
         ex_2 = ex_2.evalf()
     else:
         d_ex2 = d_ex.subs(x,x0)
         d_ex2 = d_ex2.evalf()

         d2_ex2 = d2_ex2.subs(x,x0)
         d2_ex2 = d2_ex2.evalf()

         y2 = sympify(y)   
         y = y2 - ((ex_2 * d_ex2)/ Pow(d_ex2,2) - (ex_2*d2_ex2))
         
         ex_2 = ex_2.subs(x0,y) 
         ex_2 = ex_2.evalf()
         error = Abs(y - x0)
         er = sympify(error)
         er.evalf()
         error = er
         ex = ex_2
         x0 = y
     i += 1     

 print("i : " + str(i))
 print("xi : " + str(y))
 print("f(xi) : " + str(ex))
 print("error : " + str(error))
 return   
 
def newton(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 d_ex = diff(ex, x)
 
 y = x0
 ex_2 = ex
 d_ex2 = ex

 while((error > cond) and (i < numMax)):
     if i == 0:
         ex_2 = ex.subs(x,x0)
         ex_2 = ex_2.evalf()
         d_ex2 = d_ex.subs(x,x0)
         d_ex2 = d_ex2.evalf()
     else:
         y2 = sympify(y)
         y = y2 - (ex_2/d_ex2) 
         
         ex_2 = ex.subs(x0,y)
         ex_2 = ex.evalf()
         
         d_ex2 = d_ex2.subs(x0,y)
         d_ex2 = d_ex2.evalf(x0,y)

         error = Abs(y - x0)
         er = sympify(error)
         error = er.evalf()
         print(str(error))
         ex = ex_2
         d_ex = d_ex2   
         x0 = y
     i += 1     

 print("i : " + str(i))
 print("xi : " + str(y))
 print("f(xi) : " + str(ex))
 print("error : " + str(error))
 return 

def puntoFijo(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)
 
 y = x0
 ex_2 = 0

 print("Ingrese el polinomio g(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 gx = input("Aqui : ")
 rx = sympify(gx)

 rx_2 = 0

 while((error > cond) and (i < numMax)):
     if i == 0:
         
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()

         rx_2 = rx.subs(x,y)
         rx_2 = rx_2.evalf()

     else:
         y = rx_2.evalf()
         
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
         
         rx_2 = rx.subs(x,y)
         rx_2 = rx_2.evalf()
         
         error = Abs(y - x0)
         er = sympify(error)
         error = er.evalf()
            
         x0 = y
         
         print("i : " + str(i))
         print("xi : " + str(y))
         print("f(xi) : " + str(ex_2))
         print("g(xi) : " + str(rx_2))
         print("error : " + str(error))
     i += 1     
 return

def busIncrem(x0, d, numMax):
 x = Symbol('x')
 i = 0 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 y = x0
 ex_2 = 0.1
 ex_3 = 0.1
 while (i < numMax):
     if i == 0:
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
     else:
         x0 = y
         y = y + d        
         
         ex_3 = ex_2

         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
         
         if (ex_2*ex_3 < 0):
              print("Hay una raiz de f en i: " + str(i))
              print("a : " + str(ex_3))
              print("b : " + str(ex_2))
              
     i += 1     
 return
def bisec(a, b, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 xm = 0
 xm0 = 0
 ex_2 = 0
 ex_3 = 0

 w = 0

 while (error > cond) and (i < numMax):
     if i == 0:
         xm = (a + b)/2
         
         ex_2 = ex.subs(x,xm)
         ex_3 = ex_2.evalf()
         
         ex_2 = ex.subs(x, a)
         ex_2 = ex_2.evalf()
     else:
        
         if (w < 0) :
             b = xm
         else:
             a = xm
         
         xm0 = xm
         xm = (a+b)/2
         
         ex_2 = ex.subs(x,xm)
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
     i += 1
 print("i : " + str(i))
 print("a : " + str(a))
 print("xm : " + str(xm))
 print("b : " + str(b))
 print("f(xm) : " + str(ex_3))
 print("error : " + str(error))
 return     

def reglaFalsa(a, b, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 xm = 0
 xm0 = 0
 ex_2 = 0
 ex_3 = 0
 ex_a = 0
 ex_b = 0

 while (error > cond) and (i < numMax):
     if i == 1:
         ex_2 = ex.subs(x,a)
         ex_2 = ex_2.evalf()
         ex_a = ex_2
         
         ex_2 = ex.subs(x,b)
         ex_2 = ex_2.evalf()
         ex_b = ex_2
         
         xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)
         ex_3 = ex.subs(x, xm)
         ex_3 = ex_3.evalf()
     else:
        
         if (ex_a*ex_3 < 0) :
             b = xm
         else:
             a = xm
         
         xm0 = xm
         ex_2 = ex.subs(x,a)
         ex_2 = ex_2.evalf()
         ex_a = ex_2

         ex_2 = ex.subs(x,b)
         ex_2 = ex_2.evalf()
         ex_b = ex_2

         xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)

         ex_3 = ex.subs(x, xm)
         ex_3 = ex_3.evalf()
         
         error = Abs(xm-xm0)
         er = sympify(error)
         error = er.evalf()   
     i += 1
 print("i : " + str(i))
 print("a : " + str(a))
 print("xm : " + str(xm))
 print("b : " + str(b))
 print("f(xm) : " + str(ex_3))
 print("error : " + str(error))
 return    

def secan(x0, x1, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 y = x0
 ex_0 = ex
 ex_1 = ex
 
 while((error > cond) and (i < numMax)):
     if i == 0:
         ex_0 = ex.subs(x,x0)
         ex_0 = ex_0.evalf()
     
     elif i == 1:
         ex_1 = ex.subs(x,x1)
         ex_1 = ex_1.evalf()

     else:
         y = x1
         x1 = x1 - (ex_1*(x1 - x0)/(ex_1 - ex_0))
         x0 = y
         
         ex_0 = ex_1.subs(x,x0)
         ex_0 = ex_1.evalf()
         
         ex_1 = ex.subs(x,x1)
         ex_1 = ex_1.evalf()

         error = Abs(x1 - x0)
         er = sympify(error)
         error = er.evalf()  
     i += 1     

 print("i : " + str(i))
 print("x : " + str(x1))
 print("f : " + str(ex_1))
 print("error : " + str(error))
 return

def gaussSimple(Ma, b):

    output = {
        "type": 2
        "method": "Simple Gaussian Reduction",
        }

    # Getting matrix dimention
    n = len(Ma)

    # Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))
    
    steps = {'Step 0': np.copy(M)}

    # Matrix reduction
    for i in range(n-1):
        for j in range(i+1,n):
            if (M[j,i] != 0):
                M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]
        steps[f'Step {i+1}'] = np.copy(M)
    
    output["results"] = steps

    output["x"] = backSubst(M)

    return output

def gaussPartialPivot(Ma, b):

    output = {
        "type": 2
        "method": "Gaussian Reduction With Partial Pivoting",
        }

    #Getting matrix dimention
    n = len(Ma)

    #Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))

    steps = {'Step 0': np.copy(M)}

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
        steps[f'Step {i+1}'] = np.copy(M)
    
    output["results"] = steps

    output["x"] = backSubst(M)

    return output

def gaussTotalPivot(Ma, b):

    output = {
        "type": 2
        "method": "Gaussian Reduction With Total Pivoting",
        }

    #Getting matrix dimention
    n = len(Ma)

    #Adding the the vector B at the end of the matrix
    matrixMa = np.matrix(Ma)
    vectorB = np.array(b)
    M = np.column_stack((matrixMa,vectorB))
    changes = np.array([])
    
    steps = {'Step 0': np.copy(M)}

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

def outputToString(output):
    if(output["type"]):
        return outputIncrementalSearch(output)
    elif (output["type"]):
        return outputanalyticalMethod(output)
    else:
        return outputGauss(output)

def outputIncrementalSearch(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput = "\nResults:\n"
    results = output["results"]
    for i in results:
        stringOutput += f'There is a root of f in {i}\n'
    stringOutput += "\n______________________________________________________________\n"

def outputanalyticalMethod(output):
    stringOutput = f'\n{output["method"]}\n'
    stringOutput = "\nResults table:\n"
    columns = output["columns"]
    for i in columns:
        stringOutput += '|{:15}'.format(i)
    stringOutput += "|\n"
    results = output["results"]
    for j in results:
        for k in j:
            stringOutput += '|{:15E}'.format(k)
        stringOutput += "|\n"
    if ("root" in output):
        stringOutput = f'\nAn approximation of the root was fund in: {output["root"]}\n'
    else:
        stringOutput = f'\nAn approximation of the root was not found in {output["iterations"]}\n'
    stringOutput += "\n______________________________________________________________\n"

def outputGauss(output):
    stringOutput = f'\n{output["method"]}'
    stringOutput += "\nResults:\n"
    results = output["results"]
    for i,j in results.items():
        stringOutput += f'\n{i}\n\n'
        for k in j:
            for m in k:
                stringOutput += '{:15E}'.format(m.item())
            stringOutput += "\n"
    stringOutput += "\nAfter Back Sustitution\n"
    stringOutput += "\nx:\n"
    for i in output["x"]:
        stringOutput += '{:15E}\n'.format(i.item())

    stringOutput += "\n______________________________________________________________\n"

    return stringOutput
