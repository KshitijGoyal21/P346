#Question 3 Gauss seidel with diagonal dominance included
import rand_no_lcg2 as rg
import numpy as np
import matrix_multiply as mm
def gs():
    ctr = 0
    n = []# temprory storing of numbers
    str_coeff = ''
    a = []  #input matrix LHS
    b=[]#input matrix RHS
    # Reading from file
    with open("Assignment3_Q3_data.txt") as file:
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
    if is_symmetric(a) and pos_definite(a,10):
        print("Convergence is possible")
    x = np.zeros(len(a))
    e =list(x)#matrix to stor error for each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    sum=0
    sum2=0
    print("Initial matrix", a)
    # Diagonally dominant row swap
    for i in range(0, len(a)):
        if abs(a[i][i]) < sum_row(a[i], a[i][i]):
            for j in range(0, len(a)):
                if i != j:
                    if abs(a[i][j]) >= sum_row(a[i], a[i][j]):
                        a = swap(a, i, j)
                        b = swap_b(b, i, j)
                        i = i - 1
                        break

    print("Daigonally dominant matrix", a)
    print("Matrix for LHS",b)
    while f != len(x):
        if ctr > 25:
            print("series not converging in 25 iteration")
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
                    if e[i] < 0.000001:#setting precision
                        f = f + 1
                e[i] = x[i]
    print("Output matrix")
    print(x)
    print("Number of iterations")
    print(ctr)
    #pos_definite(a,10)


def pos_definite(a, seed):#For positive definite
    x = np.zeros((len(a), 1))
    x_temp = rg.rand_no(len(a), seed)
    for i in range(0, len(a)):
        x[i][0] = x_temp[i]
    x_t = transpose(x)
    x1 = mat_mult(a, x_t)
    x2 = mat_mult(x, x1)
    while True:
        if x2[0][0] > 0:
            print("positive definite")
            return True
            break
        else:
            pos_definite(a, seed + 1)


def mat_mult(x1, x2):#matrix multiplication
    y = []
    for i in range(0, len(x1)):
        sum = 0
        for j in range(0, len(x1)):
            sum += x1[j] * x2[i]
        y.append(sum)
    return y
def swap(a,i,j):#swap for matrix a
    for k in range(0,len(a)):
        temp=a[i][k]
        a[i][k]=a[j][k]
        a[j][k]=temp
    return a
def swap_b(b,i,j):#swap for matrix b
        temp2 = b[i]
        b[i] = b[j]
        b[j]=temp2
        return b

def sum_row(x,sub):#sum of elements of row without element sub
    sum=0
    for i in range(0, len(x)):
        sum+=abs(x[i])
    sum=sum-sub
    return sum
def is_symmetric(a):#to check for symmetric matrix
    for i in range(0, len(a)):
        for j in range(i, len(a)):
            if a[i][j] != a[j][i]:
                return False
    return True
def transpose(x):#to get column vector
     x_t=np.zeros((len(x),1))
     for i in range(0,len(x)):
         x_t[i][0]=x[i]
     return x_t
gs()
#OUTPUT
# Initial matrix [[4, 0, 4, 10, 1], [0, 4, 2, 0, 1], [2, 5, 1, 3, 13], [11, 3, 0, 1, 2], [3, 2, 7, 1, 0]]
# Daigonally dominant matrix [[11, 3, 0, 1, 2], [0, 4, 2, 0, 1], [3, 2, 7, 1, 0], [4, 0, 4, 10, 1], [2, 5, 1, 3, 13]]
# Matrix for LHS [51, 15, 15, 20, 92]
# Output matrix
# [2.97916517 2.21559959 0.21128404 0.15231696 5.7150336 ]
# Number of iterations
# 13
