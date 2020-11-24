from django.shortcuts import render
from django.http import HttpResponse
from LOZA.methods import Trazlin,outputToString,TrazlinCubicos

def trazlin(request):
    return render(request, "trazlin.html")

def verTrazlin(request):
    x = request.GET["x"]
    X = x.split(",")
    X = [int(i) for i in X]
    y = request.GET["y"]
    Y = y.split(",")
    Y = [float(i) for i in Y]
    output = Trazlin(X,Y)
    Dic = outputToString(output)
    prueba = Dic.split("\n")
    TraceCof = [prueba[7], prueba[8], prueba[9]]
    Traz = [prueba[12], prueba[13], prueba[14]]
    #print(Dic)
    print(TraceCof)
    print(Traz)
    return render(request, "trazlin.html",{"coef":TraceCof, "tracers":Traz})
