from django.shortcuts import render
from LOZA import methods
import numpy as np
import numpy.matlib
from numpy.lib import scimath


def incrementalSearch(request):

    if len(request.GET)>1:
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
    error = ""
    method = request.GET["method"]
    if len(request.GET)>1:
        Ma = toMatrix(request.GET["Ma"])
        b = toVector(request.GET["b"])
        if(method == "LU With Gaussian Elimination"):
            output = preRenderLU(methods.LUSimple(Ma, b))
        elif (method == "LU With Partial Pivot"):
            output = preRenderLU(methods.LUPartialPivot(Ma, b))
        elif (method == "Crout"):
            output = preRenderLU(methods.crout(Ma, b))
        elif (method == "Doolittle"):
            output = preRenderLU(methods.doolittle(Ma, b))
        elif (method == "Cholesky"):
            output = preRenderLU(methods.cholesky(Ma, b))

    return render(request, 'lufact.html', {"method": method, "output": output, "error": error})

def gauss(request):
    output = ""
    error = ""
    method = request.GET["method"]
    if len(request.GET)>1:
        Ma = toMatrix(request.GET["Ma"])
        b = toVector(request.GET["b"])
        if(method == "Simple Gaussian Elimination"):
            output = preRenderGauss(methods.gaussSimple(Ma, b))
        elif (method == "Gaussian Elimination With Partial Pivoting"):
            output = preRenderGauss(methods.gaussPartialPivot(Ma, b))
        elif (method == "Gaussian Elimination With Total Pivoting"):
            output = preRenderGauss(methods.gaussTotalPivot(Ma, b))

    return render(request, 'gauss.html', {"method": method, "output": output, "error": error})

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
    return render(request, "trazlin.html")

def viewTrazlin(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [int(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    Type = request.GET["type"]
    if Type == "LinearTracers":
        output = Trazlin(X,Y)
        Dic = outputToString(output)
    elif Type == "QuadraticPlotters":
        output = TrazlinQuadratic(X,Y)
        Dic = outputToString(output)
    else:
        output = TrazlinCubicos(X,Y)
        Dic = outputToString(output)
    
    data = Dic.split("\n")
    TraceCof = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]

    return render(request, "trazlin.html",{"coef":TraceCof, "tracers":Traz})

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
    return render(request, "BisecReg.html")

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
    Type = request.GET["type"]
    if Type == "FalseRule":
        output = regulaFalsi(A,B,Fx,Error,N)
        Dic = outputToString(output)
    elif Type == "Secant":
        output = secan(A,B,Fx,N)
        Dic = outputToString(output)
    else:
        output = bisec(A,B,Fx,Error,N)
        Dic = outputToString(output)
    data = Dic.split("\n")
    data[len(data)-2] = ""
    data[1] = ""
    data[2] = ""
    return render(request, "BisecReg.html", {"data":output})

def NewtPoint(request):
    return render(request, "NewtonPoint.html")

def viewNewtPoint(request):
    Fx = request.GET["fx"]
    Gx = request.GET["gx"]
    X0 = request.GET["x0"]
    X0 = float(X0)
    N = request.GET["n"]
    N = int(N)
    Type = request.GET["type"]
    if Type == "Newton":
        output = newton(Fx,X0,N)
        Dic = outputToString(output)
    else:
        output = fixedPoint(Fx,Gx,X0,N)
        Dic = outputToString(output)
    return render(request, "NewtonPoint.html", {"data":output})

def RaizMul(request):
    return render(request,"RaizMul.html")

def viewRaizMul(request):
    Fx = request.GET["fx"]
    X0 = request.GET["x0"]
    X0 = float(X0)
    N = request.GET["n"]
    N = int(N)
    output = mulRoots(Fx,X0,N)
    Dic = outputToString(output)
    data = Dic.split("\n")
    data[len(data)-2] = ""
    return render(request, "RaizMul.html", {"data":output})