from sympy import *
def mulRoots(fx, x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 ex = sympify(fx)

 d_ex = diff(ex, x)
 d2_ex = diff(d_ex, x)
 
 y = x0
 ex_2 = 0
 ex_3 = 0
 d_ex2 = 0
 d2_ex2 = 0 

 while error > cond and i < numMax:
     if i == 0:
         ex_2 = ex.subs(x,x0)
         ex_2 = ex_2.evalf()
     else:
         d_ex2 = d_ex.subs(x,x0)
         d_ex2 = d_ex2.evalf()

         d2_ex2 = d2_ex.subs(x,x0)
         d2_ex2 = d2_ex2.evalf()
  
         y = y - ((ex_2 * d_ex2)/ Pow(d_ex2,2) - (ex_2*d2_ex2))
         
         ex_3 = ex.subs(x,y) 
         ex_3 = ex_3.evalf()
         
         error = Abs(y - x0)
         er = sympify(error)
         error = er.evalf()
         
         
         ex_2 = ex_3
         x0 = y
     i += 1     

 return   
 
def newton(fx, x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

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

 return 

def puntoFijo(fx, gx, x0, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

 ex = sympify(fx)
 
 y = x0
 ex_2 = 0

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
         
     i += 1     
 return

def busIncrem(fx, xo, d, numMax):
 x = Symbol('x')
 i = 0 

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
              print("identificador ")
              
     i += 1     
 return
def bisec(a, b, fx, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

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
 return     

def reglaFalsa(a, b, fx, numMax):
 x = Symbol('x')
 i = 1 
 cond = 0.0000001
 error = 1.0000000 

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
 return    

def secan(x0, x1, fx, numMax):
 x = Symbol('x')
 i = 0
 cond = 0.0000001
 error = 1.0000000    

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
 return

 
