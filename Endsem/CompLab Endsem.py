import math

import library
import library as lib
import matplotlib.pyplot as plot
import numpy as np
def Q1():
    #Reading from textfile
    a=lib.input_textfile("Q1_Endsem.txt")
    # Array on RHS is simply identity matrix
    b = np.zeros((len(a), len(a[0])))
    #Making Identity matrix for RHS
    for i in range(0, len(a)):
        for j in range(0, len(a[0])):
            if i == j:
                b[i][j] = 1
            else:
                b[i][j] = 0
    a_inv=lib.LUdecomp(a,b)
    print("Inverse is",a_inv)
    print(np.matmul(a,a_inv))
Q1()
#OUTPUT
# Inverse is
# [[-0.70794915  2.53143336  2.43119586  0.96657586 -3.90227717]
#  [-0.19343159  0.3101425   0.27946587  0.05771514 -0.29413477]
#  [ 0.02168902  0.36546521  0.28614837  0.05055532 -0.28992037]
#  [ 0.27341203 -0.12992456  0.13161265 -0.14101355  0.44885676]
#  [ 0.7815265  -2.87510478 -2.67893732 -0.70113859  4.23384092]]

def Q2():
    #Plotting number of points on the lext box as a function of time for T=200000
    lib.box2(5000,200000,1)
Q2()

def Q3():
    #Max stretch of spring is the point where Force on the spring is 0
    #Hence we need to find roots of 2.5-xexp(x)
    root=lib.newton_rhapson_root("2.5-x*math.exp(x)","-1*(1+x)*math.exp(x)")
    print("Max stretch=",root)
Q3()
#Max stretch= 0.958586356728703


def Q4():
    #Using simpson
    integration=lib.simpson_integration_N("1/((1-(math.sin(math.pi)*math.sin(x))**2)**0.5)",math.pi/2,0,10)
    #print(integration)
    integration=integration*4*math.sqrt(1/(9.8))
    print(integration)
Q4()
#OUTPUT
#2.007089923154493


def Q5():
    #Height of y without air resistance is 10^2/(2*10)=5
    #Now using RK4
    hl, vl, tl=lib.RK4_1st_2nd_deriv(0, 10,0.1, 5)
    hl_pos=[]
    vl2=[]
    for i in range(0,len(hl)):
        if hl[i]>0:
            hl_pos.append(hl[i])
            vl2.append(vl[i])
    print(hl)
    print("Max height is",max(hl))
    plot.xlabel("height")
    plot.ylabel("velocity")
    plot.plot(hl_pos,vl2)
    plot.show()
Q5()
#Max height is 4.933830677673028



def Q6():
    x = lib.input_textfile("Endsem_Q6.txt")
    e_val, e_vec, k = lib.e_val(x)
    print("Eigenvalues are", e_val)
    print("Eigenvectors are", e_vec)
    print("Number of iterations", k)
#Q6()
# Eigenvalues are [[8.00026209]]
# Eigenvectors are [array([-0.19805779]), array([0.69315443]), array([0.69304404]), array([2.94456827e-07])]
# Number of iterations 9

def Q7():
    x= lib.input_textfile("esem_fit.txt")
    #print(x)
    x_ln = []
    y_ln = []
    x_list = []
    for i in range(0, len(x)):
        x_ln.append([x[i][0]])
        y_ln.append(x[i][1])
    #print(x_ln)
    fit=lib.poly_fit(x_ln,y_ln, 4)
    #print(fit)
    #coeffiecients are
    for i in range(0,len(fit)):
        print(fit[i][len(fit[0])-1],"x^",i)
    plot.scatter(x_ln,y_ln)
    y_fit=lib.plot_polynomial(x_ln)
    plot.plot(x_ln,y_fit)
    plot.show()
Q7()
#OUTPUT
#After fitting
# 0.25462950721154803 x^ 0
# -1.193759213809225 x^ 1
# -0.4572554123829691 x^ 2
# -0.8025653910658195 x^ 3
# 0.013239427477396579 x^ 4