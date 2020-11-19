import methods

A = [
    [4,-1,0,3],
    [1,15.5,3,8],
    [0,-1.3,-4,1.1],
    [14,5,-2,30]
]

b = [1,1,1,1]

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
