from django.shortcuts import render
from LOZA import methods
import numpy as np
import numpy.matlib
from numpy.lib import scimath


def incrementalSearch(request):

    if len(request.GET)>=4:
        fx = request.GET["function"]
        x0 = float(request.GET["x0"])
        dx = float(request.GET["dx"])
        iterations = int(request.GET["iterations"])
        output = methods.incremSearch(fx, x0, dx, iterations)
        return render(request,'incrementalSearch.html', {"method": "Incremental Search", "output": zip(range(1,len(output['results'])+1), output['results']), "error": ""})
    else: 
        return render(request,'incrementalSearch.html', {"method": "Incremental Search", "output": "", "error": ""})

def LUFact(request):
    output = ""
    errors = ""
    method = request.GET["method"]
    if len(request.GET)>=3:
        Ma = toMatrix(request.GET["Ma"])
        b = toVector(request.GET["b"])
        if(method == "LU With Gaussian Elimination"):
            rawOutput = methods.LUSimple(Ma, b)
        elif (method == "LU With Partial Pivot"):
            rawOutput = methods.LUPartialPivot(Ma, b)
        elif (method == "Crout"):
            rawOutput = methods.crout(Ma, b)
        elif (method == "Doolittle"):
            rawOutput = methods.doolittle(Ma, b)
        elif (method == "Cholesky"):
            rawOutput = methods.cholesky(Ma, b)

        errors = rawOutput["errors"]
        print("errors",errors)
        if(len(errors)==0):
            output = preRenderLU(rawOutput)

    return render(request, 'lufact.html', {"method": method, "output": output, "errors": errors})

def gauss(request):
    output = ""
    errors = ""
    method = request.GET["method"]
    if len(request.GET)>=3:
        Ma = toMatrix(request.GET["Ma"])
        b = toVector(request.GET["b"])
        if(method == "Simple Gaussian Elimination"):
            rawOutput = methods.gaussSimple(Ma, b)
        elif (method == "Gaussian Elimination With Partial Pivoting"):
            rawOutput = methods.gaussPartialPivot(Ma, b)
        elif (method == "Gaussian Elimination With Total Pivoting"):
            rawOutput = methods.gaussTotalPivot(Ma, b)

        errors = rawOutput["errors"]
        if(len(errors)==0):
            output = preRenderGauss(rawOutput)

    return render(request, 'gauss.html', {"method": method, "output": output, "errors": errors})

def preRenderLU(output):
    results = output["results"]
    newOutput = dict()
    newOutput["method"] = output["method"]
    steps = dict()
    for i,j in results.items():
        auxOut = ""
        for k in j:
            if(isinstance(k,numpy.ndarray)):
                for l in k:
                    for m in l:
                        real = f'{(m.item()).real:.6f}' if not (m.item()).real == 0 else ""
                        imag = f'{(m.item()).imag:+.6f}j' if not (m.item()).imag == 0 else ""
                        num = real+imag if real or imag else f'{0:.6f}'
                        auxOut+= f'{num:^15}'
                    auxOut += "\n"
            else:
                for l,m in k.items():
                    auxOut += f'\n{l}\n'
                    for n in m:
                        for o in n:
                            real = f'{(o.item()).real:.6f}' if not (o.item()).real == 0 else ""
                            imag = f'{(o.item()).imag:+.6f}j' if not (o.item()).imag == 0 else ""
                            num = real+imag if real or imag else f'{0:.6f}'
                            auxOut+= f'{num:^15}'
                        auxOut += "\n"
        steps[i] = auxOut
    newOutput["results"] = steps

    auxX = list()
    for i in output["x"]:
        real = f'{(i.item()).real:.6f}' if not (i.item()).real == 0 else ""
        imag = f'{(i.item()).imag:+.6f}j' if not (i.item()).imag == 0 else ""
        num = real+imag if real or imag else f'{0:.6f}'
        auxX.append(f'{num:^15}\n')
    newOutput["x"]=auxX

    return newOutput

def preRenderGauss(output):
    results = output["results"]
    newOutput = dict()
    newOutput["method"] = output["method"]
    steps = dict()
    for i, j in results.items():
        auxOut = ""
        for k in j:
            for m in k:
                auxOut += '{:^25E}'.format(m.item())
            auxOut += "\n"
        steps[i] = auxOut
    newOutput["results"] = steps
    
    auxX = list()
    for i in output["x"]:
        auxX.append('{:^25E}\n'.format(i.item()))
    newOutput["x"]=auxX
    
    return newOutput

def toMatrix(matrixStr):
    matrixStr = matrixStr.replace(" ","")
    matrixStr = matrixStr.replace("\n","")
    rows = matrixStr.split(";")
    auxM = []
    for row in rows:
        splitedRow = row.split(",")
        auxR = []
        for num in splitedRow:
            auxR.append(float(num))
        auxM.append(auxR)
    return auxM

def toVector(vectorStr):
    splitedVector = vectorStr.split(",")
    auxV = list()
    for num in splitedVector:
        auxV.append(float(num))
    return auxV

            


def index(request):

    return render(request, 'application.html')
from django.shortcuts import render
from django.http import HttpResponse
from LOZA.methods import Trazlin,TrazlinQuadratic,TrazlinCubicos,outputToString,TrazlinCubicos,incremSearch,bisec,regulaFalsi,newton,fixedPoint,secan,mulRoots

def trazlin(request):
    return render(request, "trazlin.html", {"method": request.GET["method"]})

def viewTrazlin(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [float(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    Method = request.GET["method"]
    if Method == "Linear Tracers":
        output = Trazlin(X,Y)
    elif Method == "Quadratic Tracers":
        output = TrazlinQuadratic(X,Y)
    elif Method == "Cubic Tracers":
        output = TrazlinCubicos(X,Y)
    TraceCof = ""
    Traz = ""
    Errors = output["errors"]
    if(len(Errors)==0):    
        Dic = outputToString(output)
        data = Dic.split("\n")
        TraceCof = [data[7], data[8], data[9]]
        Traz = [data[12], data[13], data[14]]
        

    return render(request, "trazlin.html",{"method": Method, "coef":TraceCof, "tracers":Traz ,"errors":Errors})

def search(request):
    return render(request, "Search.html")

def viewSearch(request):
    Fx = request.GET["fx"]
    X0 = request.GET["x0"]
    X0 = float(X0)
    Delta = request.GET["delta"]
    Delta = float(Delta)
    N = request.GET["n"]
    N = int(N)
    output = incremSearch(Fx,X0,Delta,N)
    Dic = outputToString(output)
    data = Dic.split("\n")
    data[len(data)-2] = ""
    return render(request, "Search.html", {"data":data})

def BisecReg(request):
    return render(request, "BisecReg.html", {"method": request.GET["method"]})

def viewBisecReg(request):
    Fx = request.GET["fx"]
    A = request.GET["a"]
    A = float(A)
    B = request.GET["b"]
    B = float(B)
    Error = request.GET["error"]
    Error = float(Error)
    N = request.GET["n"]
    N = int(N)
    Method = request.GET["method"]
    if Method == "False Rule":
        output = regulaFalsi(A,B,Fx,Error,N)
    elif Method == "Secant":
        output = secan(A,B,Fx,Error,N)
    elif Method == "Bisection":
        output = bisec(A,B,Fx,Error,N)

    return render(request, "BisecReg.html", {"method": Method,"data":output})

def NewtPoint(request):
    return render(request, "NewtonPoint.html", {"method": request.GET["method"]} )

def viewNewtPoint(request):
    Fx = request.GET["fx"]
    X0 = request.GET["x0"]
    X0 = float(X0)
    Tol = request.GET["tol"]
    Tol = float(Tol)
    N = request.GET["n"]
    N = int(N)
    Method = request.GET["method"]
    if Method == "Newton":
        output = newton(Fx,X0,Tol,N)
        Errors = output["errors"]
        #Dic = outputToString(output)
    elif Method == "Fixed Point":
        Gx = request.GET["gx"]
        output = fixedPoint(Fx,Gx,X0,Tol,N)
        Errors = output["errors"]
        #Dic = outputToString(output)
    return render(request, "NewtonPoint.html", {"method": Method, "data":output, "errors":Errors})

def RaizMul(request):
    return render(request,"RaizMul.html")

def viewRaizMul(request):
    Fx = request.GET["fx"]
    X0 = request.GET["x0"]
    X0 = float(X0)
    N = request.GET["n"]
    N = int(N)
    Tol = request.GET["error"]
    Tol = float(Tol)
    output = mulRoots(Fx,X0,Tol,N)
    Errors = output["errors"]
    return render(request, "RaizMul.html", {"data":output, "errors": Errors})