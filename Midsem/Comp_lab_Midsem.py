#Q1 Area of ellipse
import library as lib
import math
import matplotlib.pyplot as plot
def area_ellipse(a,b):
    #Area of ellipse is math.pi*a*b
    ctr=0#to count number of points inside ellipse
    ctr2=0#to count total number of points
    for i in range(0,1000):
        rg1=lib.rand_no_LCG_pos_k(10,i,1)#random number for semi minor axis
        rg2=lib.rand_no_LCG_pos_k(15,i,2)  # random number for semi major axis
        if (rg1**2)/(a**2)+(rg2**2)/(b**2)<1:
            ctr=ctr+1
        ctr2=ctr2+1
    #Now we know that (Area of ellipse)/(Area of rectangel i.e.4ab)=(No. of points inside)/(Total number of points)
    print("Area is",ctr/ctr2*4*a*b)
    #print((ctr/ctr2*4*a*b-(math.pi*2))/(math.pi*2)*100)
area_ellipse(1,2)
#OUTPUT  area_ellipse
#No. of random number taken is 1000
#Area is 6.176

#Q2 Wein's Displacement Law
def wein():
    root=lib.newton_rhapson_root()
    b=6.626*10**(-34)*3*10**(8)/(1.381*10**(-23)*root)
    print("Wein's constant is,",b*1000,"* 10^(-3)")
wein()

#OUTPUT FOR WEIN
#Wein's constant is, 2.899010330737035 * 10^(-3)


#Q3 Using Gauss Seidel
def eq_sol():
     x,y=lib.read_file("msem_gs.txt")
     print(x)
     print(y)
     lib.gs(x,y)
eq_sol()
#OUTPUT FOR GAUSS SEIDEL Q3
#[ 1.49999983 -0.5   2.  -2.49999991  1.  -1.]



#Q4 data fit
def data_model_fit():
    x,y=lib.read_file("msem_fit.txt")
    x_ln=[]
    y_ln=[]
    x_list=[]
    for i in range(0,len(x)):
        x_ln.append(math.log(x[i][0]))
        y_ln.append(math.log(y[i]))
        x_list.append(x[i][0])
    #print(x_ln)
    #print(y_ln)
    #Now we can linearly fit for the model y=ax^b by taking ln,i.e. ln y=ln(a)+bln(x). Now we can fit ln(y) vs ln(x)
    #Similarly for y=ae^(-bx) by taking ln,i.e. ln y=ln(a)-bx. Now we can fit ln(y) vs x
    #coefficient for y=ax^b model is gievn by
    m1,c1,p_c1=lib.linear_fit_data(x_ln,y_ln,1)#1 denotes sigma squared
    # coefficient for y=ae^(-bx) model is gievn by
    m2,c2,p_c2=lib.linear_fit_data(x_list,y_ln,1)#1 denotes sigma squared
    print("Pearson coeff for y=ax^b is",p_c1)
    print("Pearson coeff for y=ae^(-bx) is",p_c2)
    print("For Eq y=ax^b, b here is,",m1,"and a here is,",math.exp(c1))
    print("For Eq y=ae^(-bx), b here is,",-1*m2, "and a here is,", math.exp(c2))

    if (1-p_c1)<(1-p_c2):
        print("Model of y=ax^b is better as its pearson coeff is closer to 1")
    else:
        print("Model of y=ae^(-bx) is better")
    #For plotting we take any two points and make the line y=mx+c
    plot.scatter(x_ln,y_ln)
    y1,y2=lib.line_plotter(m1,c1)
    plot.plot([0.9,3],[y1,y2])
    plot.show()
    plot.scatter(x_list, y_ln)
    y1, y2 = lib.line_plotter2(m2,c2)
    plot.plot([2.5, 21], [y1, y2])
    plot.show()
data_model_fit()
#OUTPUT FOR Q4
# Pearson coeff for y=ax^b is 0.7750435352872259
# Pearson coeff for y=ae^(-bx) is 0.5762426888065756
# For Eq y=ax^b, b here is, -0.53740930145056 and a here is, 21.046352159550004
# For Eq y=ae^(-bx), b here is, 0.05845553447818332 and a here is, 12.21299282456827
# Model of y=ax^b is better as its pearson coeff is closer to 1
