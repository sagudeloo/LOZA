import django.http as dj
from django.template import Template, Context
from django.shortcuts import render
from LOZA import methods
import numpy as np

def showJacobi(request):
    return render(request,"jacobi.html")

def tJacobi(request):
    x0 = request.GET["x0"]
    tol = request.GET["tol"]
    numMax = request.GET["numMax"]
    Ma = request.GET["A"]
    A = Ma.split(",")
    A = [A.insert(i,A[i].split(" ")) for i in A]
    Vb = request.GET["b"]
    b = Vb.split(" ")
    x0 = int(x0)
    tol = int(tol)
    numMax = int(numMax)
    output = methods.jacobiM(A,b,x0,tol,numMax)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]

    return render(request, "trazlin.html",{"coef":Coef, "tracers":Traz})

def showGaussSei(request):
    return render(request,"gauss_seidel.html")

def tGaussSei(request):
    x0 = request.GET["x0"]
    tol = request.GET["tol"]
    numMax = request.GET["numMax"]
    Ma = request.GET["A"]
    A = Ma.split(",")
    A = [A.insert(i,A[i].split(" ")) for i in A]
    Vb = request.GET["b"]
    b = Vb.split(" ")
    x0 = int(x0)
    tol = int(tol)
    numMax = int(numMax)
    output = methods.jacobiM(A,b,x0,tol,numMax)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]  

def showSor(request):
    return render(request,"sor.html")

def tSOR(request):
    x0 = request.GET["x0"]
    tol = request.GET["tol"]
    numMax = request.GET["numMax"]
    Ma = request.GET["A"]
    A = Ma.split(",")
    A = [A.insert(i,A[i].split(" ")) for i in A]
    Vb = request.GET["b"]
    b = Vb.split(" ")
    x0 = int(x0)
    tol = int(tol)
    numMax = int(numMax)
    output = methods.jacobiM(A,b,x0,tol,numMax)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]

def showVandermonde(request):
    return render(request,"vandermonde.html")

def tVandermonde(request):
    x0 = request.GET["x0"]
    tol = request.GET["tol"]
    numMax = request.GET["numMax"]
    Ma = request.GET["A"]
    A = Ma.split(",")
    A = [A.insert(i,A[i].split(" ")) for i in A]
    Vb = request.GET["b"]
    b = Vb.split(" ")
    x0 = int(x0)
    tol = int(tol)
    numMax = int(numMax)
    output = methods.jacobiM(A,b,x0,tol,numMax)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]

def showDifdivid(request):
    return render(request,"difdivid.html")

def tDifdivid(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [int(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    output = methods.difdivid(X,Y)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]

def showLagrange(request):
    return render(request,"Lagrange.html")

def tLagrange(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [int(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    output = methods.difdivid(X,Y)
    Dic = methods.outputToString(output)
    data = Dic.split("\n")
    Coef = [data[7], data[8], data[9]]
    Traz = [data[12], data[13], data[14]]  

def showHome(request):
     return render(request, 'application.html')      