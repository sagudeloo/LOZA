from sympy import *
def mulRoots(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 d_ex = diff(ex, x)
 d2_ex = diff(d_ex, x)
 
 y = x0
 ex_2 = ex
 d_ex2 = ex
 d2_ex2 = ex 

 while error > cond and i < numMax:
     if i == 0:
         ex_2 = ex.subs(x,x0)
         ex_2 = ex_2.evalf()
     else:
         d_ex2 = d_ex.subs(x,x0)
         d_ex2 = d_ex2.evalf()

         d2_ex2 = d2_ex2.subs(x,x0)
         d2_ex2 = d2_ex2.evalf()

         y2 = sympify(y)   
         y = y2 - ((ex_2 * d_ex2)/ Pow(d_ex2,2) - (ex_2*d2_ex2))
         
         ex_2 = ex_2.subs(x0,y) 
         ex_2 = ex_2.evalf()
         error = Abs(y - x0)
         er = sympify(error)
         er.evalf()
         error = er
         ex = ex_2
         x0 = y
     i += 1     

 print("i : " + str(i))
 print("xi : " + str(y))
 print("f(xi) : " + str(ex))
 print("error : " + str(error))
 return   
 
def newton(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 d_ex = diff(ex, x)
 
 y = x0
 ex_2 = ex
 d_ex2 = ex

 while((error > cond) and (i < numMax)):
     if i == 0:
         ex_2 = ex.subs(x,x0)
         ex_2 = ex_2.evalf()
         d_ex2 = d_ex.subs(x,x0)
         d_ex2 = d_ex2.evalf()
     else:
         y2 = sympify(y)
         y = y2 - (ex_2/d_ex2) 
         
         ex_2 = ex.subs(x0,y)
         ex_2 = ex.evalf()
         
         d_ex2 = d_ex2.subs(x0,y)
         d_ex2 = d_ex2.evalf(x0,y)

         error = Abs(y - x0)
         er = sympify(error)
         error = er.evalf()
         print(str(error))
         ex = ex_2
         d_ex = d_ex2   
         x0 = y
     i += 1     

 print("i : " + str(i))
 print("xi : " + str(y))
 print("f(xi) : " + str(ex))
 print("error : " + str(error))
 return 

def puntoFijo(x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)
 
 y = x0
 ex_2 = 0

 print("Ingrese el polinomio g(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 gx = input("Aqui : ")
 rx = sympify(gx)

 rx_2 = 0

 while((error > cond) and (i < numMax)):
     if i == 0:
         
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()

         rx_2 = rx.subs(x,y)
         rx_2 = rx_2.evalf()

     else:
         y = rx_2.evalf()
         
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
         
         rx_2 = rx.subs(x,y)
         rx_2 = rx_2.evalf()
         
         error = Abs(y - x0)
         er = sympify(error)
         error = er.evalf()
            
         x0 = y
         
         print("i : " + str(i))
         print("xi : " + str(y))
         print("f(xi) : " + str(ex_2))
         print("g(xi) : " + str(rx_2))
         print("error : " + str(error))
     i += 1     
 return

def busIncrem(x0, d, numMax):
 x = Symbol('x')
 i = 0 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 y = x0
 ex_2 = 0.1
 ex_3 = 0.1
 while (i < numMax):
     if i == 0:
         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
     else:
         x0 = y
         y = y + d        
         
         ex_3 = ex_2

         ex_2 = ex.subs(x,y)
         ex_2 = ex_2.evalf()
         
         if (ex_2*ex_3 < 0):
              print("Hay una raiz de f en i: " + str(i))
              print("a : " + str(ex_3))
              print("b : " + str(ex_2))
              
     i += 1     
 return
def bisec(a, b, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 xm = 0
 xm0 = 0
 ex_2 = 0
 ex_3 = 0

 w = 0

 while (error > cond) and (i < numMax):
     if i == 0:
         xm = (a + b)/2
         
         ex_2 = ex.subs(x,xm)
         ex_3 = ex_2.evalf()
         
         ex_2 = ex.subs(x, a)
         ex_2 = ex_2.evalf()
     else:
        
         if (w < 0) :
             b = xm
         else:
             a = xm
         
         xm0 = xm
         xm = (a+b)/2
         
         ex_2 = ex.subs(x,xm)
         ex_3 = ex_2.evalf()
         
         ex_2 = ex.subs(x, a)
         ex_2 = ex_2.evalf()

         error = Abs(xm-xm0)
         er = sympify(error)
         error = er.evalf()

         if (ex_2*ex_3 < 0):
              w = -1
         else:
             w = 1    
     i += 1
 print("i : " + str(i))
 print("a : " + str(a))
 print("xm : " + str(xm))
 print("b : " + str(b))
 print("f(xm) : " + str(ex_3))
 print("error : " + str(error))
 return     

def reglaFalsa(a, b, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

 print("Ingrese el polinomio f(x) de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 xm = 0
 xm0 = 0
 ex_2 = 0
 ex_3 = 0
 ex_a = 0
 ex_b = 0

 while (error > cond) and (i < numMax):
     if i == 1:
         ex_2 = ex.subs(x,a)
         ex_2 = ex_2.evalf()
         ex_a = ex_2
         
         ex_2 = ex.subs(x,b)
         ex_2 = ex_2.evalf()
         ex_b = ex_2
         
         xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)
         ex_3 = ex.subs(x, xm)
         ex_3 = ex_3.evalf()
     else:
        
         if (ex_a*ex_3 < 0) :
             b = xm
         else:
             a = xm
         
         xm0 = xm
         ex_2 = ex.subs(x,a)
         ex_2 = ex_2.evalf()
         ex_a = ex_2

         ex_2 = ex.subs(x,b)
         ex_2 = ex_2.evalf()
         ex_b = ex_2

         xm = (ex_b*a - ex_a*b)/(ex_b-ex_a)

         ex_3 = ex.subs(x, xm)
         ex_3 = ex_3.evalf()
         
         error = Abs(xm-xm0)
         er = sympify(error)
         error = er.evalf()   
     i += 1
 print("i : " + str(i))
 print("a : " + str(a))
 print("xm : " + str(xm))
 print("b : " + str(b))
 print("f(xm) : " + str(ex_3))
 print("error : " + str(error))
 return    

def secan(x0, x1, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 print("Ingrese el polinomio de la siguiente forma --> potencia: **, raíz: //, ejemplo: x**2; x//2.\nPara números decimales use el punto en lugar de la coma.")
 fx = input("Aqui : ")
 ex = sympify(fx)

 y = x0
 ex_0 = ex
 ex_1 = ex
 
 while((error > cond) and (i < numMax)):
     if i == 0:
         ex_0 = ex.subs(x,x0)
         ex_0 = ex_0.evalf()
     
     elif i == 1:
         ex_1 = ex.subs(x,x1)
         ex_1 = ex_1.evalf()

     else:
         y = x1
         x1 = x1 - (ex_1*(x1 - x0)/(ex_1 - ex_0))
         x0 = y
         
         ex_0 = ex_1.subs(x,x0)
         ex_0 = ex_1.evalf()
         
         ex_1 = ex.subs(x,x1)
         ex_1 = ex_1.evalf()

         error = Abs(x1 - x0)
         er = sympify(error)
         error = er.evalf()  
     i += 1     

 print("i : " + str(i))
 print("x : " + str(x1))
 print("f : " + str(ex_1))
 print("error : " + str(error))
 return

 
