from django.shortcuts import render
from django.http import HttpResponse
from LOZA.methods import Trazlin,outputToString,TrazlinCubicos,incremSearch,bisec,regulaFalsi,newton,fixedPoint,secan,mulRoots

def trazlin(request):
    return render(request, "trazlin.html")

def viewTrazlin(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [int(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    output = Trazlin(X,Y)
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
    return render(request, "BisecReg.html", {"data":data})

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
    data = Dic.split("\n")
    data[len(data)-2] = ""
    return render(request, "NewtonPoint.html", {"data":data})

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
    return render(request, "RaizMul.html", {"data":data})