import methods

def main():
    x = int(input("Enter the number of the function/method you would like to use: \n Multiple roots:1 \n Newton's method:2 \n Fixed-point iteration:3 \n Incremental search:4 \n Bisection method:5 \n Regula falsi:6 \n Secant method:7 \n Simple Gauss:8 \n Gauss partial pivot:9 \n Gauss total pivot:0 \n"))
    if (x==1):
        fx = input("Enter the function f(x) ")
        x0=input("Enter the initial value of x")
        x0 = float(x0)
        numMax=input("Enter the max. number of iterations")
        numMax = int(numMax)
        output = methods.mulRoots(fx,x0,numMax)
        print(methods.outputToString(output))
    elif x==2:
        fx = input("Enter the function f(x) ")
        x0=input("Enter the initial value of x")
        x0 = float(x0)
        numMax=input("Enter the max. number of iterations")
        numMax = int(numMax)
        output = methods.newton(fx, x0, numMax)
        print(methods.outputToString(output))
    elif x==3:
        fx = input("Enter the function f(x) ")
        x0=input("Enter the initial value of x ")
        x0 = float(x0)
        numMax=input("Enter the max. number of iterations ")
        numMax = int(numMax)
        gx = input("Enter the function g(x) ")
        output = methods.fixedPoint(fx, gx, x0, numMax)
        print(methods.outputToString(output))
    elif x==4:
        fx = input("Enter the function f(x) ")
        x0=input("Enter the initial value of x ")
        x0 = float(x0)
        numMax=input("Enter the max. number of iterations ")
        numMax = int(numMax)
        d = input("Enter the delta value ")
        d = float(d)
        output = methods.incremSearch(fx, x0, d,numMax)
        print(methods.outputToString(output))
    elif x==5:
        fx = input("Enter the function f(x) ")
        a = input("Enter the initial value of a ")
        a = float(a)
        b = input("Enter the initial value of b ")
        b = float(b)
        numMax=input("Enter the max. number of iterations ")
        numMax = int(numMax)
        output = methods.bisec(a, b, fx, numMax)
        print(methods.outputToString(output))
    elif x==6:
        fx = input("Enter the function f(x) ")
        a = input("Enter the initial value of a ")
        a = float(a)
        b = input("Enter the initial value of b ")
        b = float(b)
        numMax=input("Enter the max. number of iterations ")
        numMax = int(numMax)
        output = methods.regulaFalsi(a,b,fx,numMax)
        print(methods.outputToString(output))
    elif x==7:
        fx = input("Enter the function f(x) ")
        x0 = input("Enter the x0 value of the series")
        x0 = float(x0)
        x1 = input("Enter the x1 value of the series")
        x1 = float(x1)
        numMax=input("Enter the max. number of iterations ")
        numMax = int(numMax)
        output = methods.secan(x0,x1,fx,numMax)
        print(methods.outputToString(output))
    elif x==8:
        n = int(input("Enter the n dimension of the matrix: "))
        print("Enter the A matrix the following way: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Enter the b vector the following way: 5 6")
        b = readVector(n)
        output = methods.gaussSimple(A, b)
        print(methods.outputToString(output))
    elif x==9:
        n = int(input("Enter the n dimension of the matrix: "))
        print("Enter the A matrix the following way: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Enter the b vector the following way: 5 6")
        b = readVector(n)
        output = methods.gaussPartialPivot(A, b)
        print(methods.outputToString(output))
    elif x==0:
        n = int(input("Enter the n dimension of the matrix: "))
        print("Enter the A matrix the following way: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Enter the b vector the following way: 5 6")
        b = readVector(n)
        output = methods.gaussTotalPivot(A, b)
        print(methods.outputToString(output))
    else:
        print("Nonexistent o wrongly typed number.")

def readMatrix(n):
    matrix = list()
    for i in range(n):
        row = list()
        stringRow = input()
        stringRow = stringRow.split(" ")
        for j in range(n):
            row.append(float(stringRow[j]))
        matrix.append(row)
    return matrix

def readVector(n):
    vector = list()
    stringVector = input()
    stringVector = stringVector.split(" ")
    for i in range(n):
        vector.append(float(stringVector[i]))
    return vector


if __name__ == "__main__":
    main()
