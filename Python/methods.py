import numpy as np
import numpy.matlib


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
