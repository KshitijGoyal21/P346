#Question 3 LU decomposition
import numpy as np
def LUdecomp():
    a=[]#LHS matrix
    b=[]#RHS matrix
    ctr=0
    str_coeff = ''#used to take input from file
    n = []  # temprory storing of numbers
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
                    sign = -1#deciding sign
                else:
                    sign = 1
                if t >= 48 and t <= 57:
                    if i < len(content) - 1:
                        if ord(content[i + 1]) >= 48 and ord(content[i + 1]) <= 57:
                            str_coeff = str_coeff + content[i]
                            continue
                        str_coeff = str_coeff + content[i]
                        if i==len(content)-2:
                            b.append(sign * int(str_coeff))#making array
                        else:
                            n.append(sign * int(str_coeff))

                str_coeff = ""
            a.append(n)#making array
            n = []
            ctr = ctr + 1
    print("Input array on LHS")
    print(a)
    print("Input array on RHS")
    print(b)
    #LU decomposition
    for j in range(0, len(a)):
        for i in range(1, j + 1):
            sum = 0
            for k in range(0, i):
                sum = sum + a[i][k] * a[k][j]
            a[i][j] = a[i][j] - sum
        for i in range(j + 1, len(a)):
            sum2 = 0
            for k in range(0, j):
                sum2 = sum2 + a[i][k] * a[k][j]
            a[i][j] = (a[i][j] - sum2) / a[j][j]
    print("Input array on RHS after LU decomp")
    print(a)
    # Now forward backward
    for i in range(1, len(a)):
        sum = 0
        for j in range(0, i):
            sum = sum + a[i][j] * b[j]
        b[i] = b[i] - sum
    for i in range(len(a) - 1, -1, -1):
        sum2 = 0
        for j in range(i + 1, len(a)):
            sum2 = sum2 + a[i][j] * b[j]
        b[i] = (b[i] - sum2) / (a[i][i])
    print("Output array")
    print(b)


LUdecomp()
# OUTPUT
# Input array on LHS
# [[4, 0, 4, 10, 1], [0, 4, 2, 0, 1], [2, 5, 1, 3, 13], [11, 3, 0, 1, 2], [3, 2, 7, 1, 0]]
# Input array on RHS
# [20, 15, 92, 51, 15]
# Input array on RHS after LU decomp
# [[4, 0, 4, 10, 1], [0.0, 4.0, 2.0, 0.0, 1.0], [0.5, 1.25, -3.5, -2.0, 11.25], [2.75, 0.75, 3.5714285714285716, -19.357142857142858, -41.67857142857143], [0.75, 0.5, -0.8571428571428571, 0.4243542435424354, 26.079335793357934]]
# Output array
# [2.9791651927838654, 2.2155995755217557, 0.2112840466926055, 0.15231694375663485, 5.715033604527767]
