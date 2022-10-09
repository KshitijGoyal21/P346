#Question 2 Cholesky
import rand_no_lcg2 as rg
import numpy as np
import matrix_multiply as mm
import math
def cholesky():
    ctr = 0
    n = []# temprory storing of numbers
    str_coeff = ''
    a = []  #input array LHS
    b=[]#input array RHS
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
                sign = -1
            else:
                sign = 1
            if t >= 48 and t <= 57:
                if i < len(content) - 1:
                    if ord(content[i + 1]) >= 48 and ord(content[i + 1]) <= 57:
                        str_coeff = str_coeff + content[i]
                        continue
                    str_coeff = str_coeff + content[i]
                    if i == len(content) - 2:
                        b.append(sign * int(str_coeff))
                    else:
                        n.append(sign * int(str_coeff))

            str_coeff = ""
        a.append(n)
        n = []
        ctr = ctr + 1
    if pos_definite(a,10):#condition for cholesky
        print("Inital matrix on LHS",a)
        print("Inital matrix on RHS", b)
        for i in range(0, len(a)):
            sum = 0
            for j in range(0, i):
                sum = sum + a[i][j] ** 2
            a[i][i] = (a[i][i] - sum) ** (0.5)
            for j in range(i + 1, len(a)):
                sum2 = 0
                for k in range(0, i):
                    sum2 = sum2 + a[i][k] * a[k][j]
                a[j][i] = (1 / a[i][i]) * (a[i][j] - sum2)
                a[i][j] = a[j][i]
        print("Matrix after cholesky")
        print(a)
        # Now forward backward
        for i in range(0, len(a)):
            sum = 0
            for j in range(0, i):
                sum = sum + a[i][j] * b[j]
            b[i] = (b[i] - sum) / a[i][i]
        for i in range(len(a) - 1, -1, -1):

            sum2 = 0
            for j in range(i + 1, len(a)):
                sum2 = sum2 + a[i][j] * b[j]
            b[i] = (b[i] - sum2) / (a[i][i])
        print("Output matrix")
        print(b)
    else:
        print("matrix is not positive definite")



def pos_definite(a,seed):#positive definite
     x = np.zeros((len(a),1))
     x_temp=rg.rand_no(len(a),seed)
     for i in range(0,len(a)):
         x[i][0]=x_temp[i]
     x_t=transpose(x)
     x1=mat_mult(a,x_t)
     x2=mat_mult(x,x1)
     while True:
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

cholesky()
# \OUTPUT
# positive definite
# Inital matrix on LHS [[4, -1, 0, -1, 0, 0], [-1, 4, -1, 0, -1, 0], [0, -1, 4, 0, 0, -1], [-1, 0, 0, 4, -1, 0], [0, -1, 0, -1, 4, -1], [0, 0, -1, 0, -1, 4]]
# Inital matrix on RHS [2, 1, 2, 2, 1, 2]
# Matrix after cholesky
# [[2.0, -0.5, 0.0, -0.5, 0.0, 0.0], [-0.5, 1.9364916731037085, -0.5163977794943222, -0.12909944487358055, -0.5163977794943222, 0.0], [0.0, -0.5163977794943222, 1.9321835661585918, -0.034503277967117704, -0.13801311186847082, -0.5175491695067657], [-0.5, -0.12909944487358055, -0.034503277967117704, 1.9318754766140744, -0.5546053999849018, -0.009243423333081693], [0.0, -0.5163977794943222, -0.13801311186847082, -0.5546053999849018, 1.8457244010396843, -0.5832696492049564], [0.0, 0.0, -0.5175491695067657, -0.009243423333081693, -0.5832696492049564, 1.841698654119145]]
# Output matrix
# [1.0, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0]

