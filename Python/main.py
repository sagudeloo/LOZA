import methods

def main():
    x = int(input("Ingrese el número de desea utilizar: \n Raíces múltiples:1 \n Método de newton:2 \n Método de punto fijo:3 \n Búsquedas incrementales:4 \n Bisección:5 \n Regla falsa:6 \n Secante:7 \n Gauss Simple:8 \n Gauss partial pivot:9 \n Gauss total pivot:0 \n"))
    if (x==1):
        fx = input("Ingrese la funcion f(x) ")
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        output = methods.mulRoots(fx,x0,numMax)
        print(methods.outputToString(output))
    elif x==2:
        fx = input("Ingrese la funcion f(x) ")
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        output = methods.newton(fx, x0, numMax)
        print(methods.outputToString(output))
    elif x==3:
        fx = input("Ingrese la funcion f(x) ")
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        gx = input("Ingrese la función g(x)")
        output = methods.puntoFijo(fx, gx, x0, numMax)
        print(methods.outputToString(output))
    elif x==4:
        fx = input("Ingrese la funcion f(x) ")
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        d = input("Ingrese el valor delta")
        d = float(d)
        output = methods.busIncrem(fx, x0, d,numMax)
        print(methods.outputToString(output))
    elif x==5:
        fx = input("Ingrese la funcion f(x) ")
        a = input("Ingrese el valor a")
        a = float(a)
        b = input("Ingrese el valor b")
        b = float(b)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        output = methods.bisec(a, b, fx, numMax)
        print(methods.outputToString(output))              
    elif x==6:
        fx = input("Ingrese la funcion f(x) ")
        a = input("Ingrese el valor a")
        a = float(a)
        b = input("Ingrese el valor b")
        b = float(b)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        output = methods.reglaFalsa(a,b,fx,numMax)
        print(methods.outputToString(output))
    elif x==7:
        fx = input("Ingrese la funcion f(x) ")
        x0 = input("Ingrese el valor inicial de la serie")
        x0 = float(x0)
        x1 = input("Ingrese el valor número uno de la serie")
        x1 = float(x1)
        numMax = input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        output = methods.secan(x0,x1,fx,numMax)
        print(methods.outputToString(output))
    elif x==8:
        n = int(input("Ingrese la dimension n de la matriz: "))
        print("Ingrese la matriz A de la siguiente forma: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Ingrese el vector b de la siguiente forma: 5 6")
        b = readVector(n)
        output = methods.gaussSimple(A, b)
        print(methods.outputToString(output))
    elif x==9:
        n = int(input("Ingrese la dimension n de la matriz: "))
        print("Ingrese la matriz A de la siguiente forma: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Ingrese el vector b de la siguiente forma: 5 6")
        b = readVector(n)
        output = methods.gaussPartialPivot(A, b)
        print(methods.outputToString(output))
    elif x==0:
        n = int(input("Ingrese la dimension n de la matriz: "))
        print("Ingrese la matriz A de la siguiente forma: \n 1 2\n 3 4")
        A = readMatrix(n)
        print("Ingrese el vector b de la siguiente forma: 5 6")
        b = readVector(n)
        output = methods.gaussTotalPivot(A, b)
        print(methods.outputToString(output))
    else:
        print("Método inexistente o número mal digitado.")

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
