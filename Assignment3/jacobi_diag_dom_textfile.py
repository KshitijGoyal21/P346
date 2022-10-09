import numpy as np
def jacobi():
    ctr = 0
    n = []
    str_coeff = ''
    a = []  # temprory storing of numbers
    b = []
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
    x = np.zeros(len(a))#inital matrix element
    e = list(x)#matrix to store error of each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    print("Initial matrix",a)
    #Diagonally dominant row swap
    for i in range(0, len(a)):
        if abs(a[i][i])<sum_row(a[i],a[i][i]):
            for j in range(0,len(a)):
                if i!=j:
                    if abs(a[i][j])>=sum_row(a[i],a[i][j]):
                       a=swap(a,i,j)
                       b=swap_b(b,i,j)
                       i=i-1
                       break

    print("Daigonally dominant matrix",a)
    print("Matrix for RHS",b)
    while f != len(a):
        if ctr > 25:
            break
        else:
            f = 0
            ctr = ctr + 1
            for i in range(0, len(x)):
                sum = 0
                for j in range(0, len(x)):
                    if i != j:
                        sum = sum + a[i][j] * x[j]
                x[i] = (1 / a[i][i]) * (b[i] - sum)
                if e[i] != 0:

                    e[i] = abs(-e[i] + x[i]) / abs(e[i])
                    # print(e[i])
                    if e[i] < 0.000001:
                        f = f + 1
                e[i] = x[i]
    print("Output array")
    print(x)
    print("Number of iteration is", ctr)

def swap(a,i,j):
    for k in range(0,len(a)):
        temp=a[i][k]
        a[i][k]=a[j][k]
        a[j][k]=temp
    return a
def swap_b(b,i,j):
        temp2 = b[i]
        b[i] = b[j]
        b[j]=temp2
        return b

def sum_row(x,sub):
    sum=0
    for i in range(0, len(x)):
        sum+=abs(x[i])
    sum=sum-sub
    return sum
jacobi()
# #OUTPUT
# Initial matrix [[4, 0, 4, 10, 1], [0, 4, 2, 0, 1], [2, 5, 1, 3, 13], [11, 3, 0, 1, 2], [3, 2, 7, 1, 0]]
# Daigonally dominant matrix [[11, 3, 0, 1, 2], [0, 4, 2, 0, 1], [3, 2, 7, 1, 0], [4, 0, 4, 10, 1], [2, 5, 1, 3, 13]]
# Matrix for RHS [51, 15, 15, 20, 92]
# Output array
# [2.97916517 2.21559959 0.21128404 0.15231696 5.7150336 ]
# Number of iteration is 13