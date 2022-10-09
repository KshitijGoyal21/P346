#Question 2 Gauss seidel
import rand_no_lcg2 as rg
import numpy as np
def gs():
    ctr = 0
    n = []# temprory storing of numbers
    str_coeff = ''#used to take input from file
    a = []  #Input Matrix for LHS
    b=[]#Input Matrix for RHS
    # Reading from file
    with open("Assignment3_Q2_data.txt") as file:
      while True:

        content = file.readline()
        content = content.replace(" ", ",")  # removing all spaces from the line for easier reading
        if not content:
            break
        for i in range(0, len(content), 1):
            t = ord(content[i])  # ord funtions takes character and returns its ascii value
            if ord(content[i - 1]) == 45:
                sign = -1#deciding sign
            else:
                sign = 1
            if t >= 48 and t <= 57:
                if i < len(content) - 1:
                    if ord(content[i + 1]) >= 48 and ord(content[i + 1]) <= 57:
                        str_coeff = str_coeff + content[i]
                        continue
                    str_coeff = str_coeff + content[i]
                    if i == len(content) - 2:
                        b.append(sign * int(str_coeff))#making array
                    else:
                        n.append(sign * int(str_coeff))

            str_coeff = ""
        a.append(n)#making array
        n = []
        ctr = ctr + 1
    if is_symmetric(a) and pos_definite(a,10):
        print("Convergence is possible")
    #Also possible if strictly diagonally dominant
    x = np.zeros(len(a))#inital value matrix
    e =list(x)#array to store error for each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    sum=0
    sum2=0
    print("Input array on LHS")
    print(a)
    print("Input array on RHS")
    print(b)
    #Gauss seidel iteration
    while f != len(x):
        if ctr > 25:
            print("series not converging in 25 iteration")#limiting number of iterations
            break
        else:
            f = 0
            ctr = ctr + 1
            for i in range(0, len(x)):
                sum = 0
                sum2= 0

                for j in range(0,i):# remember that we start here form 0 and go to i not i-1 because in that formula the index of matrix satrt from 1 and hence i starts from 1
                        sum2 = sum2 + a[i][j] * x[j]
                for j in range(i+1, len(x)):
                        sum = sum + a[i][j] * x[j]

                x[i] = (1 / a[i][i]) * (b[i] - sum2 - sum)
                if e[i] != 0:

                    e[i] = abs(-e[i] + x[i]) / abs(e[i])
                    # print(e[i])
                    if e[i] < 0.000001:#10^-6 precision
                        f = f + 1
                e[i] = x[i]
    print("Output array")
    print(x)
    print("Number of iteration is",ctr)



def pos_definite(a,seed):
     x = np.zeros((len(a),1))
     x_temp=rg.rand_no(len(a),seed)
     for i in range(0,len(a)):
         x[i][0]=x_temp[i]
     x_t=transpose(x)
     x1=mat_mult(a,x_t)
     x2=mat_mult(x,x1)
     while True:#taking random matrix and iteration until condition for positive definite is met
      if x2[0][0]>0:
         print("positive definite")
         return True
         break
      else:
         pos_definite(a, seed + 1)
def mat_mult(x1,x2):#matrix multiplication
    y=[]
    for i in range(0,len(x1)):
        sum=0
        for j in range(0,len(x1)):
            sum+=x1[j]*x2[i]
        y.append(sum)
    return y
def transpose(x):#to get column vector
     x_t=np.zeros((len(x),1))
     for i in range(0,len(x)):
         x_t[i][0]=x[i]
     return x_t
def is_symmetric(a):
    for i in range(0, len(a)):
        for j in range(i, len(a)):
            if a[i][j] != a[j][i]:
                return False
    return True
gs()
# positive definite
# Input array on LHS
# [[4, -1, 0, -1, 0, 0], [-1, 4, -1, 0, -1, 0], [0, -1, 4, 0, 0, -1], [-1, 0, 0, 4, -1, 0], [0, -1, 0, -1, 4, -1], [0, 0, -1, 0, -1, 4]]
# Input array on RHS
# [2, 1, 2, 2, 1, 2]
# Output array
# [0.99999975 0.99999979 0.99999991 0.99999985 0.99999987 0.99999995]
# Number of iteration is 16
