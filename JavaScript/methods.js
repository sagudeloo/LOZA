math = require("./node_modules/mathjs/dist/math.js");

class methods {

    static busquedasIncrementales(fx,x0,delta,maxNum){
        console.log("Busquedas Incrementales");
        let x=x0;
        let i=0;
        let f = 0.1;
        let f0 = 0.1;
        while(i < maxNum){
            if(i == 0){
                f = math.evaluate(fx);
                f = f(x);
                f0 = f;
            }else{
                x0 = x;
                x = x + delta;
                f0 = f;
                f = math.evaluate(fx);
                f = f(x); 
            }
            if (f*f0 < 0 ){
                console.log("Hay una raiz de f en [" + x0+","+x+"]");
            }
            i = i+1;
        }

    }

    static biseccion(fx,a,b,maxNum){
        console.log("Biseccion");
        
        let cond = 0.0000001;
        let i = 1;
        let xm = 0;
        let xm0 = 0;
        let fa = 0;
        let fm = 0;
        let check = 0;
        let error = 1;

        console.log("| i  |     a         |    xm         |      b        |        f(x)          |    E    |");

        while(error > cond && i < maxNum){
            //Primera iteracion
            
            if(i == 1){
                xm = (a+b)/2;
                fa = math.evaluate(fx);
                fm = fa(xm)
                fa = fa(a);
            }else{
                //Condicion de a y b
                if(check < 0){
                    b = xm;
                }else{
                    a = xm;
                }

                xm0 = xm
                xm = (a+b)/2;
                fa = math.evaluate(fx);
                fm = fa(xm)
                fa = fa(a); 
                error = math.abs(xm-xm0);
            }
            //Valor de check
            if(fm*fa < 0){
                check = -1;
            }else{
                check = 1;
            }
            console.log("|  "+ i +" | "+a.toFixed(10)+"  | "+xm.toFixed(10)+"  | "+b.toFixed(10)+"  | "+fm+"  |    "+error+"    |");
            i = i+1;
        }
    }

    static reglaFalsa(fx,a,b,maxNum){
        console.log("Regla Falsa");

        let cond = 0.0000001;
        let i = 1;
        let error = 1;
        let xm = 0;
        let xm0 = 0;
        let f = 0;
        let fa = 0;
        let fb = 0;
        let fm = 0;

        console.log("| i  |     a         |    xm         |     b         |            f(x)         |    E    |");

        while(error > cond && i < maxNum){
            if (i == 1){
                f = math.evaluate(fx);
                fa = f(a);
                fb = f(b);
                xm = (fb*a-fa*b)/(fb-fa);
                fm = f(xm);
            }else{
                if (fa* fm < 0){
                    b = xm;
                }else{
                    a = xm;
                }
                xm0 = xm;
                fa = f(a);
                fb = f(b);
                xm = (fb*a-fa*b)/(fb-fa);
                error = math.abs(xm-xm0);
            }
            console.log("|  "+ i +" | "+a.toFixed(10)+"  | "+xm.toFixed(10)+"  | "+b.toFixed(10)+"  | "+fm+"  |    "+error+"    |");
            i = i+1;
        }
    }

    static puntoFijo(fx,gx,x0,maxNum){
        console.log("Punto Fijo");

        let cond = 0.0000001;
        let error = 1;
        let i = 0;
        let x = x0;
        let f = 0;
        let g = 0;

        console.log("| i  |          xi          |       g(xi)          |         f(xi)       |                E              |");

        while(error > cond && i < maxNum){
            if(i == 0){
                f = math.evaluate(fx);
                f = f(x);
                g = math.evaluate(gx);
                g = g(x);
                console.log("|  "+ i +" | "+x.toFixed(16)+"  | "+g+"  | "+f+"  |    "+error.toFixed(16)+"         |    ");
            }else{
                x = g;
                f = math.evaluate(fx);
                f = f(x);
                g = math.evaluate(gx);
                g = g(x);
                error = math.abs(x-x0);
                x0 = x;
                console.log("|  "+ i +" | "+x+"  | "+g+"  | "+f+"  |    "+error+"         |    ");
            }
            
            i = i+1;
        }
    }

    static newton (x0,fx,dx,maxNum){
        console.log("Newton");

        let error = 1;
        let i = 0;
        let cond = 0.0000001;
        let f = 0;
        let fdx = 0;
        let x = x0;

        console.log("| iter |         xi          |          f(x)        |          E          |");

        while(error > cond && i < maxNum){
            if (i == 0){
                f = math.evaluate(fx);
                f = f(x0);
                fdx = math.evaluate(dx);
                fdx = fdx(x0);
                console.log("|  "+ i +"   | "+x.toFixed(16)+"  | "+f+"  | "+error.toFixed(16)+"  |");
            }else{
                x = x0-(f/fdx);
                f = math.evaluate(fx);
                f = f(x);
                fdx = math.evaluate(dx);
                fdx = fdx(x);
                error=math.abs(x-x0);
                x0 = x;
                console.log("|  "+ i +"   | "+x+"  | "+f+"  | "+error+"  |");
            }
            
            i = i+1;
        }
    }

    static secante (x0,x1,fx,maxNum){
        console.log("Secante");

        let error = 1;
        let i = 0;
        let cond = 0.0000001;
        let f0 = 0;
        let f1 = 0;
        let x = x0;

        console.log("| iter |         xi          |          f(x)       |          E          |");

        while(error > cond && i < maxNum){
            if (i == 0 ){
                f0 = math.evaluate(fx);
                f0 = f0(x0);
                console.log("|  "+ i +"   | "+x.toFixed(16)+"  | "+f1.toFixed(16)+"  | "+error.toFixed(16)+"  |");
            }else if(i == 1){
                f1 = math.evaluate(fx);
                f1 = f1(x1);
                console.log("|  "+ i +"   | "+x1.toFixed(16)+"  | "+f1+"  | "+error.toFixed(16)+"  |");
            }else{
                x = x1;
                x1 = x1-f1*(x1-x0)/(f1-f0);
                x0 = x;
                f0 = f1;
                f1 = math.evaluate(fx);
                f1 = f1(x1);
                error=math.abs(x1-x0);
                console.log("|  "+ i +"   | "+x1+"  | "+f1+"  | "+error+"  |");
            }
            
            i = i+1;
            
        }
    }
    
    static raicesMultiples(fx, fdx, f2dx, x0, maxNum){
        console.log("Raices Multiples");
        
        let cond = 0.0000001;
        let x = x0;
        let i = 0;
        let error = 1;
        let f = 0;
        let fd = 0;
        let f2d = 0;
        let fac = 0;

        console.log("| i  |          xi           |         f(x)        |          E          |");

        while(error > cond && i < maxNum){
            if(i == 0){
                f = math.evaluate(fx);
                f = f(x0);
                console.log("|  "+ i +" | "+x.toFixed(18)+"  | "+f+"  | "+error.toFixed(16)+"  |");
            }else{
                fd = math.evaluate(fdx);
                fd = fd(x0);
                f2d = math.evaluate(f2dx);
                f2d = f2d(x0);
                x = x-f*fd/(math.pow(fd,2)-f*f2d);
                fac = math.evaluate(fx);
                fac = fac(x);
                error = math.abs(x-x0);
                f = fac;
                x0 = x;
                console.log("|  "+ i +" | "+x+"  | "+f+"  | "+error+"  |");
            }
            
            i = i+1;
        }
    }

}

//Busquedas Incrementales 
methods.busquedasIncrementales('f(x) = log(sin(x)^2+1)-(1/2)',-3,0.5,100);
//Biseccion
methods.biseccion('f(x) = log(sin(x)^2+1)-(1/2)',0,1,100);
//Regla Falsa
methods.reglaFalsa('f(x) = log(sin(x)^2+1)-(1/2)',0,1,100);
//Punto Fijo
methods.puntoFijo('f(x) = log(sin(x)^2+1)-(1/2)-x','f(x) = log(sin(x)^2+1)-(1/2)',-0.5,100);
//Newton
methods.newton(0.5,'f(x) = log(sin(x)^2+1)-(1/2)', 'f(x) = 2*(sin(x)^2+1)^-1*sin(x)*cos(x)',100);
//Secante
methods.secante(0.5,1,'f(x) = log(sin(x)^2+1)-(1/2)',100);
//Raices Multiples
methods.raicesMultiples('f(x) = exp(x)-x-1','f(x) = exp(x)-1','f(x) = exp(x)',1,6);