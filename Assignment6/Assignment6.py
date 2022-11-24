import library as lib
import matplotlib.pyplot as plt
def Q1():
    y,v,t = lib.RK4_1st_2nd_deriv(2, -1, 0.1, 100)
    plt.plot(t, y,label="y vs t")
    plt.plot(t, v, color = 'green',label="v vs t")
    plt.xlabel("t")
    plt.legend()
    plt.show()
Q1()

def Q2():
    x,y=lib.boundary_value_corrected()
    plt.plot(x, y,label="T vs position")
    for i in range(0, len(y)):
        if abs(y[i] - 100) < 0.1:#to get value of position at temperature 100
            output = x[i]
            break
    print("x=",output)
    plt.xlabel("position")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()
Q2()
#OUTPUT
#x= 4.42999999999995

def Q3():
    lib.PDE_sol(2,100,4,80000,0,5000)
    plt.xlabel("position")
    plt.ylabel("Temperature")
    plt.show()
Q3()

def Q4():
    e_val,e_vec,k=lib.e_val("Assignment6_Q4.txt")
    print("Eigenvalues are",e_val)
    print("Eigenvectors are", e_vec)
    print("Number of iterations", k)
Q4()
#OUTPUT
# Eigenvalues are [[3.99909952]]
# Eigenvectors are [array([0.70710678]), array([9.91799746e-05]), array([0.70710677])]
# Number of iterations 13