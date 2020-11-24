from django.shortcuts import render
from LOZA import methods

def incrementalSearch(request):

    print(len(request.GET))
    if len(request.GET)>1:
        print("hola HP")
        fx = request.GET["function"]
        x0 = float(request.GET["x0"])
        dx = float(request.GET["dx"])
        iterations = int(request.GET["iterations"])
        output = methods.incremSearch(fx, x0, dx, iterations)
        print("output", output)
        return render(request,'incrementalSearch.html', {"method": "Incremental Search", "output": zip(range(1,len(output['results'])+1), output['results'])})
    else: 
        return render(request,'incrementalSearch.html', {"method": "Incremental Search", "output": ""})

def LUFact(request):

    if len(request.GET)>0:
        method = request.GET["method"]
        if(method == "lusimple"):
            Ma = request.GET()
            b = 
            output = methods.LUSimple()
            return render(request, 'lufact.html', {"methos": output["method"], "output": output})
        elif (method == "lupartial"):
            Ma = request.GET()
            b = 
            output = methods.LUPartialPivot()
            return render(request, 'lufact.html', {"methos": output["method"], "output": output})
        elif (method == "crout"):
            pass
        elif (method == "doolittle"):
            pass
        elif (method == "cholesky"):
            pass

def toMatrix(matrixStr):
    rows = matrixStr.split("\n")


def index(request):

    return render(request, 'application.html')