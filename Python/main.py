import methods

def main(fx):
    x = input("Ingrese el número de desea utilizar: \n Raíces múltiples:1 \n Método de newton:2 \n Método de punto fijo:3 \n Búsquedas incrementales:4 \n Bisección:5 \n Regla falsa:6 \n Secante:7 \n ")
    if (x=="1"):
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        methods.mulRoots(fx,x0,numMax)
    elif x=="2":
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        methods.newton(fx, x0, numMax)
    elif x=="3":
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        gx = input("Ingrese la función g(x)")
        methods.puntoFijo(fx, gx, x0, numMax)
    elif x=="4":
        x0=input("Ingrese el valor inicial de x")
        x0 = float(x0)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        d = input("Ingrese el valor delta")
        d = float(d)
        methods.busIncrem(fx, x0, d,numMax)
    elif x=="5":
        a = input("Ingrese el valor a")
        a = float(a)
        b = input("Ingrese el valor b")
        b = float(b)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        methods.bisec(a, b, fx, numMax)              
    elif x=="6":
        a = input("Ingrese el valor a")
        a = float(a)
        b = input("Ingrese el valor b")
        b = float(b)
        numMax=input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        methods.reglaFalsa(a,b,fx,numMax)
    elif x=="7":
        x0 = input("Ingrese el valor inicial de la serie")
        x0 = float(x0)
        x1 = input("Ingrese el valor número uno de la serie")
        x1 = float(x1)
        numMax = input("Ingrese el número máximo de iteraciones")
        numMax = int(numMax)
        methods.secan(x0,x1,fx,numMax)
    else:
        print("Método inexistente o número mal digitado.")

main("log(sin(x)**2+1)-(1/2)")