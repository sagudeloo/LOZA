import LOZA.methods as methods
import numpy as np

A = [
    [4,-1,0,3],
    [1,15.5,3,8],
    [0,-1.3,-4,1.1],
    [14,5,-2,30]
]

b = [1,1,1,1]

x0 = np.zeros(4)

tol = 0.0000001

Nmax = 100

w = 1.5

X = np.array([-1, 0, 3, 4])
X2 = [-1, 0, 3, 4]

Y = np.array([15.5, 3, 8, 1])
Y2 = [15.5, 3, 8, 1]

output = methods.LUSimple(A, b)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.LUPartialPivot(A, b)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.LUPartialPivot(A, b)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.LUPartialPivot(A, b)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.LUPartialPivot(A, b)
outputToString = methods.outputToString(output)
print(outputToString)

output = methods.jacobiM(A,b,x0,tol,Nmax)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.gaussSei(A,b,x0,tol,Nmax)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.sorM(A,b,x0,w,tol,Nmax)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.vanderMon(X2,Y2)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.difdivid(X2,Y2)
outputToString = methods.outputToString(output)
print(outputToString)

output = methods.Lagrange(X,Y)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.Trazlin(X,Y)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.TrazlinQuadratic(X,Y)
outputToString = methods.outputToString(output)
print(outputToString)
output = methods.TrazlinCubicos(X,Y)
outputToString = methods.outputToString(output)
print(outputToString)