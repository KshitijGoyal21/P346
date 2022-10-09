#Question 3 LU decomposition
import numpy as np
def LUdecomp():
    a=[]#input matrix LHS
    b=[]#input matrix RHS
    ctr=0
    str_coeff = ''#used to take input from file
    n = []  # temprory storing of numbers
    # Reading from file
    with open("GJ_assignment3.txt") as file:
        while True:

            content = file.readline()
            content = content.replace(" ", "")  # removing all spaces from the line for easier reading
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
    print("Input array on RHS")
    print(a)
    print("Input array on LHS")
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
# Input array on RHS
# [[1, -1, 4, 0, 2, 9], [0, 5, -2, 7, 8, 4], [1, 0, 5, 7, 3, -2], [6, -1, 2, 3, 0, 8], [-4, 2, 0, 5, -5, 3], [0, 7, -1, 5, 4, -2]]
# Input array on LHS
# [19, 2, 13, -7, -9, 2]
# Input array on RHS after LU decomp
# [[1, -1, 4, 0, 2, 9], [0.0, 5.0, -2.0, 7.0, 8.0, 4.0], [1.0, 0.2, 1.4, 5.6, -0.6000000000000001, -11.8], [6.0, 1.0, -14.285714285714286, 76.0, -28.571428571428573, -218.57142857142858], [-4.0, -0.4, 10.857142857142858, -0.6973684210526315, -7.210526315789473, 16.28947368421055], [0.0, 1.4, 1.2857142857142856, -0.15789473684210523, 1.5172054223149114, -51.65432742440045]]
# Output array
# [-1.761817043997862, 0.8962280338740133, 4.051931404116158, -1.6171308025395421, 2.041913538501913, 0.15183248715593525]
