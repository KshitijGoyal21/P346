import sys
import numpy as np
import math
import matplotlib.pyplot as plot
#Gauss jordan with augumented matrix and returns the the whole matrix itself.
# Then we need to extract the needed part i.e. last row in most Q nd the n*n matrix for inverse
def Gauss_Jordan(n):
    #Gausss Jordan
    for i in range(0,len(n)):
        if n[i][i] == 0:
            k = max(n, i)
            n = swap_row(n, i, k)

        n = multiply(n, i, 1/n[i][i])
        #print(n)
        for j in range(0, len(n)):
            if n[j][i] != 0 and i != j:
                n = add_multi(n, j, -n[j][i], i)
                #print(n)
    return n
#swaps row such that none of the diagonals is zero
def swap_row_non_zero_diag(n):
    A=[]
    for i in range(0,len(n)):
        if n[i][i]==0:
            for j in range(0,len(n)):
                if n[j][i]!=0 and n[i][j]!=0:
                    n=swap_row(n,i,j)
                    A.append([i,j])

    return n,A
#gives max absolute value in a row
def max(n, i):
    max = 0
    k = -1
    for j in range(i,len(n)):
        if max < abs(n[j][i]):
            max = abs(n[j][i])
            k = j
    return k

#swapping i th row with kth in array n
def swap_row(n, i, k):
    for j in range(0,len(n)+1):#we have +1 here as len(a) gives numbre of rows and number of columns in augumented matrix is
                                # len(a)+1 and we need to perform the specific operation on the other part also
        temp = n[i][j]
        n[i][j] = n[k][j]
        n[k][j] = temp
    return n

#multiply ith row by constant m
def multiply(n, i, m):
    for j in range(0, len(n)+1):
        n[i][j] = n[i][j] * m
    #print(n[i],"mult",m)
    return n

#add a multiplied row i to other row j with constant c
def add_multi(n, j, c, i):
    for k in range(0, len(n)+1):
        n[j][k] = n[j][k] + c * n[i][k]
    return n
#Gauss jordan for svd
def Gauss_Jordan_svd(n):
    #Gausss Jordan
    for i in range(0,len(n)):
        if n[i][i] == 0:
            k = max(n, i)
            n = swap_row(n, i, k)
        print(n)
        if n[i][i]<0.00000001:#then approx 0
            for k in range(0,len(n)):#setting values approx 0 to 0
              if n[k][k] < 0.0000000001:
                n[k][k]=0
            return n
        n = multiply(n, i, 1/n[i][i])
        #print(n)
        for j in range(0, len(n)):
            if n[j][i] != 0 and i != j:
                n = add_multi(n, j, -n[j][i], i)
                #print(n)
    return n


#General input textfile, can use anywhere with float input
def input_textfile(filename):
    with open(filename) as f:
        array = [row.split() for row in f]
    for i in range(0,len(array)):
        for j in range(0,len(array[0])):
            array[i][j]=float(array[i][j])
    return array

#Now for inverse
def Gauss_Jordan_inv(n):
    #Gausss Jordan
    print(len(n))
    for i in range(0,len(n)):
        if n[i][i] == 0:
            k = max_inv(n, i)
            n = swap_row_inv(n, i, k)
        n = multiply_inv(n, i, 1/n[i][i])
        #print(n)
        for j in range(0, len(n)):
            if n[j][i] != 0 and i != j:
                n = add_multi_inv(n, j, -n[j][i], i)
                #print(n)
    return n
#gives max absolute value in a row
def max_inv(n, i):
    max = 0
    k = -1
    for j in range(i,len(n)):
        if max < abs(n[j][i]):
            max = abs(n[j][i])
            k = j
    return k

#swapping i th row with kth in array n for inverse i.e.len(a)+2 intead of len(a)+1
def swap_row_inv(n, i, k):
    for j in range(0,len(n)*2):#we have +1 here as len(a) gives numbre of rows and number of columns in augumented matrix is
                                # len(a)*2 and we need to perform the specific operation on the identity(initially) part also
        temp = n[i][j]
        n[i][j] = n[k][j]
        n[k][j] = temp
    return n

#multiply ith row by constant m for inverse i.e.len(a)+2 intead of len(a)+1
def multiply_inv(n, i, m):
    for j in range(0, len(n)*2):
        n[i][j] = n[i][j] * m
    #print(n[i],"mult",m)
    return n

#add a multiplied row i to other row j with constant cfor inverse i.e.len(a)+2 intead of len(a)+1
def add_multi_inv(n, j, c, i):
    print(n[j],"initial",n[i])
    for k in range(0, len(n)*2):
        n[j][k] = n[j][k] + c * n[i][k]
    print(n[j],"add_mult",c,"row", i, "added to",j)

    return n
#General file read with sign and decimal but returns array a as 2d and b as 1d
#Use read_file_complex for input, real also
def read_file(file_name):
    ctr = 0
    n = []
    str_coeff = ''  # used to take input from file
    x = []  # temporary storing of numbers
    y = []
    sign=1
    # Reading from file
    with open(file_name) as file:
        while True:

            content = file.readline()
            content = content.replace(" ", ",")  # removing all spaces from the line for easier reading
            if not content:
                break
            for i in range(0, len(content), 1):
                t = ord(content[i])  # ord funtions takes character and returns its ascii value
                if ord(content[i - 1]) == 45:
                    sign = -1  # deciding sign
                elif ord(content[i - 1]) == 43 or ord(content[i - 1]) == ",":#positive sign or a space
                    sign = 1
                if t==46:
                    str_coeff = str_coeff + content[i]
                    continue
                if (t >= 48 and t <= 57):
                    if i < len(content) - 1:
                        if (ord(content[i + 1]) >= 48 and ord(content[i + 1]) <= 57) or ord(content[i + 1])==46:
                            str_coeff = str_coeff + content[i]
                            continue
                        str_coeff = str_coeff + content[i]
                        if i == len(content) - 2:
                            y.append(sign * float(str_coeff))  # making array
                        else:
                            n.append(sign * float(str_coeff))

                str_coeff = ""
            x.append(n)  # making array
            n = []
            ctr = ctr + 1
    return x,y
#Augumented matrix read
#input matrix cannot contain only numbers, if it does then change only content = content.replace(" ", ",") else give input in the form of GJ_Assignment3.txt i.e. with variables
#Use read_file_complex for input, real also
def read_file_aug(file_name):
    ctr = 0
    n=[]
    str_coeff=''#temporary string to read from file
    a=[]#temprory storing of numbers

    #Reading from file
    with open(file_name) as file:
        while True:

            content = file.readline()
            content = content.replace(" ", ",")  # removing all spaces from the line for easier reading
            if not content:
                break
            sign = 1
            for i in range(0, len(content), 1):
                t = ord(content[i])  # ord funtions takes character and returns its ascii value
                if ord(content[i - 1]) == 45:
                    sign = -1  # deciding sign
                elif ord(content[i - 1]) == 43 or ord(content[i - 1]) == ",":#positive sign or a space
                    sign = 1
                if t==46:
                    str_coeff = str_coeff + content[i]
                    continue
                if (t >= 48 and t <= 57):
                    if i < len(content) - 1:
                        if (ord(content[i + 1]) >= 48 and ord(content[i + 1]) <= 57) or ord(content[i + 1])==46:
                            str_coeff = str_coeff + content[i]
                            continue
                        str_coeff = str_coeff + content[i]
                    n.append(sign * float(str_coeff))

                str_coeff = ""
            a.append(n)  # making array
            n=[]
            ctr = ctr + 1
    return a
#print(read_file_aug("GJ_assignment3.txt"))
#Guass seidel with positive definite and is_symmetric with accuracy 10^-6 and diagonal dominance
def gs(a,b):
    x = np.zeros(len(a))
    e = list(x)  # matrix to stor error for each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    sum = 0
    sum2 = 0
    #print("Initial matrix", a)
    # Diagonally dominant row swap
    for i in range(0, len(a)):
        if abs(a[i][i]) < sum_row(a[i], a[i][i]):
            for j in range(0, len(a)):
                if i != j:
                    if abs(a[i][j]) >= sum_row(a[i], a[i][j]):
                        a = swap_row(a, i, j)
                        b = swap_b(b, i, j)
                        i = i - 1
                        break

    #print("Daigonally dominant matrix", a)
    #print("Matrix for LHS", b)
    while f != len(x):
        if ctr > 25:
            print("series not converging in 25 iteration")
            break
        else:
            f = 0
            ctr = ctr + 1
            for i in range(0, len(x)):
                sum = 0
                sum2 = 0

                for j in range(0,
                               i):  # remember that we start here form 0 and go to i not i-1 because in that formula the index of matrix satrt from 1 and hence i starts from 1
                    sum2 = sum2 + a[i][j] * x[j]
                for j in range(i + 1, len(x)):
                    sum = sum + a[i][j] * x[j]

                x[i] = (1 / a[i][i]) * (b[i] - sum2 - sum)
                if e[i] != 0:

                    e[i] = abs(-e[i] + x[i]) / abs(e[i])
                    # print(e[i])
                    if e[i] < 0.000001:
                        f = f + 1
                e[i] = x[i]
    print("Output matrix")
    print(x)
    return x
    #print("Number of iterations",ctr)


#positive deifinte used in gauss seidel
def pos_definite(a,seed):
     x = np.zeros((len(a),1))
     x_temp=rand_no_LCG_arr(len(a),seed)
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
#Matrix multiplication for n*n matrix and column matrix only
def mat_mult(x1,x2):
    y=[]
    for i in range(0,len(x1)):
        sum=0
        for j in range(0,len(x1)):
            sum+=x1[j]*x2[i]
        y.append(sum)
    return y
#transpose of a matrix
def transpose(x):#to get column vector
     x_t=np.zeros((len(x),1))
     for i in range(0,len(x)):

         x_t[i][0]=x[i]
     return x_t
#to check if a matrix is symmetric
def is_symmetric(a):
    for i in range(0, len(a)):
        for j in range(i, len(a)):
            if a[i][j] != a[j][i]:
                return False
    return True
def jacobi(a,b):
    #print("Initial matrix",a)
    #Diagonally dominant row swap
    x = np.zeros(len(a))  # inital matrix element
    e = list(x)  # matrix to store error of each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    for i in range(0, len(a)):
        if abs(a[i][i])<sum_row(a[i],a[i][i]):
            for j in range(0,len(a)):
                if i!=j:
                    if abs(a[i][j])>=sum_row(a[i],a[i][j]):
                       a=swap_row(a,i,j)
                       b=swap_b(b,i,j)
                       i=i-1
                       break

    #print("Daigonally dominant matrix",a)
    #print("Matrix for RHS",b)
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
#swap funtin for 1d array
def swap_b(b,i,j):
        temp2 = b[i]
        b[i] = b[j]
        b[j]=temp2
        return b
#sum of elements of row except sub i.e. sum_row-sub
def sum_row(x,sub):
    sum=0
    for i in range(0, len(x)):
        sum+=abs(x[i])
    sum=sum-sub
    return sum
#matrix multiplication for any general dimension of n and m
def matrix_mult(n,m):
    q =len(n)
    w =len(n[0])
    r = len(m)
    c = len(m[0])
    if(w!=r):
        print("Multiplication not possible") #Condition for valid multiplication
        sys.exit(0)

    x=np.empty([q, c])
    sum=0
    for k in range(0,q):

      for i in range(0,c):
          for j in range(0,r):
            sum=sum+(n[k][j]*m[j][i]) #Multiplication
          x[k][i]=sum
          sum=0 #Reinitializing
    #print(x)
    return x
#cholesky with forward backward and positive definite
def cholesky(a,b):
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
#for two matrix x and y, it returns the value of y at x=a
def interpol(x,y,a):
    n=len(x)
    sum=0
    for i in range(0,n):
        pro=1
        for k in range(0,n):
            if k!=i:
                pro=pro*(a-x[k])/(x[i]-x[k])
        pro=pro*y[i]
        sum=sum+pro
    print(sum)
#Laguerre_poly
#gives us the solution of any polynomial eq by successively dividing by (x-a) where  coeff of x has to be 1
#input is of the form eg a=[6,1,-7,-1,1] where lowest value is for lowest coeff i.e for x^4-x^3-7x^2+x+6
def Laguerre_poly_root(a):
    e = 0.0001  # epsilon
    d=0.0001
    c_prev=0
    c=-30 #intial guess
    ctr=0
    while len(a)!=2:
      c=c+1
      #print(a)
      while abs(c-c_prev)>e:
         #print(c)
         c_prev = c
         #print(a)
         if (poly(a,c))<d:
             print(c)
             a=deflate(a,c)
         else:
             #print(a,len(a))
             G=derivative(a,c)/poly(a,c)
             H=G**2-deri2(a,c)/poly(a,c)
             if G<0:
                 o=(len(a)-1)/(G-((len(a)-1-1)*((len(a)-1)*H-G**2))**0.5)
             else:
                 o =(len(a)-1)/(G+((len(a)-1-1)*((len(a)-1)*H-G**2))**0.5)
             c=c-o
    print(-a[0]/a[1])
#gives the polynomial function here
def poly(a,x):
    sum=0
    for i in range(0,len(a)):
        sum=sum+a[i]*x**(i)
    #print("sum",sum)
    return sum
#basically reduces the degree of polynomial by 1 each time as it divides by (x-a) but remeber coeff of a has to be 1
def deflate(a,c):
    b=[]
    b.append(a[len(a)-1])
    for i in range(len(a) - 2, -1, -1):
        if i==0:
            break
        if i==len(a)-2:
            b.insert(0,(a[i]+a[len(a)-1]*c))
        else:
            b.insert(0,(a[i]+c*b[0]))
    return b
#First order derivative
def derivative(a,x):
    sum=0
    for i in range(0,len(a)):
        if i!=0 and x!=0:
         sum=sum+a[i]*i*x**(i-1)
    return sum
#second order derivative
def deri2(a,x):
    sum = 0
    for i in range(0,len(a)):
        if i != 1 and x!=0:
            sum = sum + a[i] * i*(i-1) * x**(i - 2)
    return sum
#LU decomposition with forward backward
def LUdecomp(a,b):
    # print("Input array on LHS")
    # print(a)
    # print("Input array on RHS")
    # print(b)
    # #LU decomposition
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
    #print("Input array on RHS after LU decomp")
    #print(a)
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
    #print("Output array")
    return b
#It can give us the root of any function eg.x-2cosx=0 etc just write the function in fn method
def newton_rhapson_root(fn,fn_der):
    e = 0.00001  # epsilon
    d=0.00001  #delta
    c_prev=-9999#to store previous
    c=5#initial guess
    ctr=0
    while abs(c-c_prev)>e or fn_nr(c,fn)>d:
         c_prev = c
         ctr=ctr+1
         c=c-(fn_nr(c,fn)/fn_derivative_nr(c,fn_der))
    #print(ctr)
    #print(c)
    #print(fn(c))
    return c
#write the function which is the input here
def fn_nr(x,fn):
    y=eval(fn)
    return y
#write the derivative of function which is the input here
def fn_derivative_nr(x,fn_der):
    y=eval(fn_der)
    return y

#polynomial data fitting
#for fitting the given x and y data using a polynomial line
#x is x axis alues, y is y axis values and k is the degrree of polynomial that we want to fit
#i.e. if we want to fit a+bx+cx^2 then k=2 and the function return a,b and c i.e. coeff of fitted data
def poly_fit(x,y,k):
    cal_x=[]
    n=[]#temporary array to store sub array
    for i in range(0,k+1):
        for j in range(i,k+i+1):
           n.append(sum_pow_k_x(j,x))
        cal_x.append(n)
        n=[]
    cal_y=[]
    for j in range(0,k+1):
        cal_y.append(sum_pow_k_y(j,y,x))
    #we cannot use jacobi or guass seidel as for positive vsalues,
    # the array is incresing and we wont get diagonal dominace hance we use gauss jordan
    #Making augumented matrix for Gauss jordan
    for i in range(0,len(cal_x)):
        cal_x[i].append(cal_y[i])
    out=Gauss_Jordan(cal_x)
    return out


#summation of kth power of all elements of a 2d matrix
def sum_pow_k_x(k,x):
    sum=0
    for i in range(0, len(x)):
        sum=sum+x[i][0]**k
    return sum
#summation of kth power of all elements of a 1d matrix
def sum_pow_k_y(k,y,x):
    sum=0
    for i in range(0, len(x)):
        sum=sum+(x[i][0]**k)*y[i]
    return sum
# LCG random number between 0 and 1 generator where x is seed
#remeember to keep seed constant every time and we should get next value of sequence
#To do this send the previous random number obtained as the seed for next iteration in the program which uses random no and change
def rand_no_LCG(x):
    # x axis values
    a = 1103515245
    c = 12345
    m = 32768
    # corresponding y axis values
    return ((a*x+c)%m)/m

#GIvesgives the kth random number in the sequence of a perticular seed x but requires last random number as input
#Used in box problem
def rand_no_prev(prev,scale):
    a = 1103515245
    c = 12345
    m = 32768
    return ((a*prev+c)%m)/m*scale
#LCG random number, use this in program
#gives the kth random number in the sequence of a perticular seed x
#sclae is the scaling factor i.e. returns value from -scale to scale
def rand_no_LCG_pos_k(x,k,scale):
    # x axis values
    a = 1103515245
    c = 12345
    m = 32768
    # corresponding y axis values
    for i in range(0,k):
       x=((a*x+c)%m)
    return x/m*scale
# LCG random number generator where x is seed
#returns array of random numbers of length iter and seed x
def rand_no_LCG_arr(iter,x):#x is the seed
    a = 1103515245
    c = 12345
    m = 32768
    y=[]
    for i in range(0,iter):
       y.append((((a*x+c)%m)/m))
       x=((a*x+c)%m)
    #print(y)
    return y

#random walk with seed and no of steps, goes to negative values also
def random_walk(iter,seed):#iter is the number of iterations

    cor1= rand_no_LCG_arr(iter,seed)#taking random numbers in an array for x coordinates
    cor2= rand_no_LCG_arr(iter,seed+10)#taking second array of random numbers for y coordinates
    sum1=0
    sum2=0
    x=[]#to store x coordinate
    y=[]#to store x coordinate
    rms=0
    for i in range(0,iter):
       x.append(sum1)
       y.append(sum2)
       sum1=sum1+(cor1[i]*((-1)**(round(cor1[i]*100))))#taking the sum and choosing the sign randomly
       sum2=sum2+(cor2[i]*((-1)**(round(cor2[i]*60))))#taking the sum and choosing the sign randomly
       rms=rms+cor1[i]**2+cor2[i]**2#to calculate random value
    #print("rms value is",(rms/(iter))**0.5)#Calculating rms
    #print("Net displacement is",(sum1**2+sum2**2)**(0.5))#Calculating net displacement
    plot.plot(x,y)
    plot.show() #showing plot
#regula falsi to calucate root of polynomial in the range of x between a and b, update e and d to change accuracy
#bracketing included,if you want to apply bracketing, then str="bracketing"
#here fn is function and is sent as eg. "(math.log(x/2)-math.sin(5*x/2))"
#a is lower and b is upper
def root_reg_fal(a, b,str,fn):
    e = 0.000001  # epsilon
    d=0.000001  #delta
    if str == "bracketing":
      a,b=bracketing_regula(a,b,fn)
      #print(a,b)
    if fn_reg_fal(a,fn) * fn_reg_fal(b,fn) > 0:
      print("For regula falsi,No root in this interval, use bracketing")
    else:

     #print(a,"  ",b)
     c_prev=0
     c=999
     ctr=0
     while abs(c-c_prev)>e or fn_reg_fal(c,fn)>d:

         c_prev = c
         ctr=ctr+1
         if fn_reg_fal(a,fn)*fn_reg_fal(b,fn)<0:
             c=b-((b-a)*fn_reg_fal(b,fn))/(fn_reg_fal(b,fn)-fn_reg_fal(a,fn))
             if fn_reg_fal(a,fn)*fn_reg_fal(c,fn)<0:
                 b=c
             else:
                 a=c
         else:
             if abs(fn_reg_fal(a,fn))>abs(fn_reg_fal(b,fn)):
                  b=b-1.5*(b-a)
             else:
                  a=a+1.5*(b-a)
         #print(a,"   count=",ctr)
         #print(c,"psp")
     #print(ctr)
     return(c)#root
     #print(fn_reg_fal(c))#has to be close to zero
#input function to regula falsi
def fn_reg_fal(x,fn):
    #print(x)
    y=eval(fn)
    return y
def bracketing_regula(a,b,fn):
    while fn_reg_fal(a,fn) * fn_reg_fal(b,fn) > 0:#check bracketing
        if abs(fn_reg_fal(a,fn)) > abs(fn_reg_fal(b,fn)):

            b = 1.2 *(b - a)
        else:
            a =1.2 *(b - a)
    return a,b
#bisection to calucate root of polynomial in the range of x between a and b, update e and d to change accuracy
#change e and d to increse accuracy
#bracketing included
#bracketing included,if you want to apply bracketing, then str="bracketing"
#here fn is function and is sent as eg. "(math.log(x/2)-math.sin(5*x/2))"
#a is lower, b is upper
def fn_bisection(a, b,str,fn):
    e = 0.000001  # epsilon for x value accuracy
    d=0.000001  #delta for y value accuracy
    #print(a,"  ",b)
    ctr=0
    if str == "bracketing":
      a,b=bracketing_bisection(a,b,fn)
    if fn2(a,fn) * fn2(b,fn) > 0:
      print("For bisection,No root in this interval, use bracketing")
    else:
     #print(a,b)
     while abs(b-a)>e or fn2(a,fn)>d:
         ctr=ctr+1
         if fn2(a,fn)*fn2(b,fn)<0:
             c=(a+b)/2
             if fn2(a,fn)*fn2(c,fn)<0:
                 b=c
             else:
                 a=c
         else:
             if abs(fn2(a,fn))>abs(fn2(b,fn)):
                  b=b-1.5*(b-a)
             else:
                  a=a+1.5*(b-a)
         #print( "count=",ctr,"     ",(a+b)/2)
     return a
     #print(a) #value of root
     #print(fn2(a))#value of y at root
     #print(abs(b-a))# final difference between a and b
     #print(ctr) #interations
#input function to bisection
def fn2(x,fn):
    #print(x)
    y=eval(fn)
    return y
#bracketing with variables a and b
def bracketing_bisection(a,b,fn):
    while fn2(a,fn) * fn2(b,fn) > 0:#check bracketing
        if abs(fn2(a,fn)) > abs(fn2(b,fn)):

            b = 1.2 *(b - a)
        else:
            a =1.2 *(b - a)
    return a,b
#returns coefficient for linear least square fit
#a2 is m and a1 is c in y=mx+c and error in value of a1 and a2 is given by err_a1 and err_a2
#We can use sig_2 as 1
def linear_fit_data(x,y,sig_2):
    S=S_x=S_y=S_yy=S_xx=S_xy=0
    for i in range(0,len(x)):
        S+=1/sig_2
        S_x+=x[i]/sig_2
        S_xx+=x[i]**2/(sig_2)
        S_y+=y[i]/sig_2
        S_xy+=x[i]*y[i]/sig_2
        S_yy+=y[i]**2/(sig_2)
    delta=S*S_xx-S_x**2
    a1=(S_xx*S_y-S_x*S_xy)/delta
    a2=(S_xy*S-S_x*S_y)/delta
    err_a1=S_xx/delta
    err_a2=S/delta
    #print("Equation is y=",a2,"x",a1)
    #print("Error in slope is",err_a1)
    #print("Error in intercept is", err_a2)
    P_c=S_xy**2/(S_xx*S_yy)#Pearson coeff
    #print("Pearson coefficient is",P_c)
    return a2,a1,P_c#a2 is slope and a1 is inetrcept
#returns the inverse of the input matrix a using gauss jordan
#Does not check determinant so add that
def matrix_inverse(a):
    b=np.zeros([len(a),len(a)])
    for i in range(0,len(a)):
        for j in range(0, len(a)):
            if i==j:
               a[i].append(1)
            else:
               a[i].append(0)
    #print(a)
    a=Gauss_Jordan_inv(a)
    #print(a)
    for i in range(0,len(a)):
        for j in range(0, len(a)):
            b[i][j]=(a[i][len(a)+j])
    return b
#Returns two points on a line to plot it
def line_plotter(m,c):
    y1=m*0.9+c
    y2=m*3+c
    return y1,y2
def line_plotter2(m,c):
    y1=m*2.5+c
    y2=m*21+c
    return y1,y2
#plotting any function from a to b limits with N steps and fn is input function
def to_plot(a,b,N,fn):
    x=np.linspace(a,b,N*N+1)
    print(x)
    y=[]
    for i in range(0,N*N+1):
        y.append(fn_to_plot(fn,x[i]))
    plot.plot(x,y)
    plot.show()
def fn_to_plot(fn,x):
    return eval(fn)
#Forward euler
#a is inital and b is final
#ini_y is value of y at x(a) i.e initial condition
#h is step size
def forward_DE(fn,a,b,ini_y,h):
   y=[ini_y]
   ctr=0
   N=int(abs(b-a)/h)#Total number of steps
   x=np.linspace(a,b,N+1)
   for i in range(0,N):
      y.append(y[i]+h*fn_forward_DE(fn,a+h*i))
   plot.scatter(x,y)
   plot.show()
   return x, y
def fn_forward_DE(fn,x):
   z=eval(fn)
   return z
#a is inital and b is final
#ini_y is value of y at x(a) i.e initial condition
#h is step size
# e.g input RK2("y-x**2+1",0,5,0.5,0.1)
#change solution function in fn_RK4_DE_sol_plot
#plots actual solution(given in fn_RK2_DE_sol_plot) and the output
def RK2(fn,a,b,ini_y,h):
    y = [ini_y]
    N = int(abs(b - a) / h)  # Total number of steps
    x = np.linspace(a, b, N + 1)
    for i in range(0, N):
        k1=h*fn_RK2_DE(fn, a + h * i,y[i])
        k2=h*fn_RK2_DE(fn,a+h*i+h/2,y[i]+k1/2)
        y.append(y[i] + k2)
    plot.scatter(x, y)
    x1 = np.linspace(a, b, N * N + 1)
    y1 = []
    for i in range(0, N * N + 1):
        y1.append(fn_RK2_DE_sol_plot(x1[i]))
    print(y1)
    plot.plot(x1,y1)
    plot.show()
def fn_RK2_DE(fn,x,y):
    return eval(fn)
def fn_RK2_DE_sol_plot(x):
    fn="(x+1)**2-math.exp(x)/2"
    return eval(fn)

#a is inital and b is final
#ini_y is value of y at x(a) i.e initial condition
#h is step size
# e.g input RK4("y-x**2+1",0,5,0.5,0.1)
#change solution function in fn_RK4_DE_sol_plot
#plots actual solution(given in fn_RK4_DE_sol_plot) and the output
def RK4(fn,a,b,ini_y,h):
    y = [ini_y]
    N = int(abs(b - a) / h)  # Total number of steps
    x = np.linspace(a, b, N + 1)
    for i in range(0, N):
        k1=h*fn_RK4_DE(fn, a + h * i,y[i])
        k2=h*fn_RK4_DE(fn,a+h*i+h/2,y[i]+k1/2)
        k3=h*fn_RK4_DE(fn,a+h*i+h/2,y[i]+k2/2)
        k4=h*fn_RK4_DE(fn,a+h*i+h,y[i]+k3)
        y.append(y[i] + 1/6*(k1+2*k2+2*k3+k4))
    plot.scatter(x, y)
    x1 = np.linspace(a, b, N * N + 1)
    y1 = []
    for i in range(0, N * N + 1):
        y1.append(fn_RK4_DE_sol_plot(x1[i]))
    print(y1)
    plot.plot(x1,y1)
    plot.show()
def fn_RK4_DE(fn,x,y):
    return eval(fn)
def fn_RK4_DE_sol_plot(x):
    fn="(x+1)**2-math.exp(x)/2"
    return eval(fn)

#Make for SHO also..................................
def coupled_3eq_RK4(fn_x,fn_y,fn_z,a,b,ini_x,ini_y,ini_z,h):
    y = [[ini_x],[ini_y],[ini_z]]
    N = int(abs(b - a) / h)  # Total number of steps
    for i in range(0, N):
        k1x=h*fn_RK4_DE_coup_3d(fn_x, y[0][i],y[1][i],y[2][i])
        k1y = h * fn_RK4_DE_coup_3d(fn_y,y[0][i], y[1][i], y[2][i])
        k1z = h * fn_RK4_DE_coup_3d(fn_z,y[0][i], y[1][i],y[2][i])

        k2x=h*fn_RK4_DE_coup_3d(fn_x,y[1][i]+k1x/2,y[1][i]+k1y/2,y[2][i]+k1z/2)
        k2y = h * fn_RK4_DE_coup_3d(fn_y,y[0][i]+k1x/2, y[1][i] + k1y/2, y[2][i] + k1z / 2)
        k2z = h * fn_RK4_DE_coup_3d(fn_z,y[0][i] + k1x / 2,y[1][i] + k1y / 2,y[2][i] + k1z/2)

        k3x=h*fn_RK4_DE_coup_3d(fn_x,y[0][i]+k2x/2,y[1][i]+k2y/2,y[2][i]+k2z/2)
        k3y = h * fn_RK4_DE_coup_3d(fn_y,y[0][i]+k2x/2, y[1][i] + k2y / 2, y[2][i] + k2z / 2)
        k3z = h * fn_RK4_DE_coup_3d(fn_z,y[0][i] + k2x / 2, y[1][i] + k2y / 2,y[2][i] + k2z/ 2)

        k4x=h*fn_RK4_DE_coup_3d(fn_x,y[0][i]+k3x,y[1][i]+k3y,y[2][i]+k3z)
        k4y = h * fn_RK4_DE_coup_3d(fn_y,y[0][i]+k3x, y[1][i] + k3y, y[2][i] + k3z)
        k4z = h * fn_RK4_DE_coup_3d(fn_z,y[0][i]+k3x,y[1][i]+k3y, y[2][i] + k3z)


        y[0].append(y[0][i] + 1/6*(k1x+2*k2x+2*k3x+k4x))
        y[1].append(y[1][i] + 1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y))
        y[2].append(y[2][i] + 1 / 6 * (k1z + 2 * k2z + 2 * k3z + k4z))
    ax=plot.axes(projection='3d')
    ax.plot3D(y[0],y[1],y[2])
    plot.show()
def fn_RK4_DE_coup_3d(fn,x,y,z):
    return eval(fn)

#For SHO

def coupled_2eq_RK4(fn_x,fn_y,a,b,ini_x,ini_y,h):
    y = [[ini_x],[ini_y]]
    N = int(abs(b - a) / h)  # Total number of steps
    for i in range(0, N):
        k1x=h*fn_RK4_DE_coup_2eq(fn_x, y[0][i],y[1][i])
        k1y = h * fn_RK4_DE_coup_2eq(fn_y,y[0][i], y[1][i])

        k2x=h*fn_RK4_DE_coup_2eq(fn_x,y[1][i]+k1x/2,y[1][i]+k1y/2)
        k2y = h * fn_RK4_DE_coup_2eq(fn_y,y[0][i]+k1x/2, y[1][i] + k1y/2)

        k3x=h*fn_RK4_DE_coup_2eq(fn_x,y[0][i]+k2x/2,y[1][i]+k2y/2)
        k3y = h * fn_RK4_DE_coup_2eq(fn_y,y[0][i]+k2x/2, y[1][i] + k2y / 2)

        k4x=h*fn_RK4_DE_coup_2eq(fn_x,y[0][i]+k3x,y[1][i]+k3y)
        k4y = h * fn_RK4_DE_coup_2eq(fn_y,y[0][i]+k3x, y[1][i] + k3y)


        y[0].append(y[0][i] + 1/6*(k1x+2*k2x+2*k3x+k4x))
        y[1].append(y[1][i] + 1 / 6 * (k1y + 2 * k2y + 2 * k3y + k4y))
    return y[0],y[1]
def fn_RK4_DE_coup_2eq(fn,x,y):
    return eval(fn)

#PDE solution
#explicit
#lx is upper limit of x
#lt is upper limit of t
#assuming lower limit is 0 in both cases
#lower_x is lower limit of x
#step_arr prints plot for the step values given here
#eg. input step_arr=[0,10,20,50,100,200,500,1000] and calling eg. PDE_sol(2,50,4,5000,step_arr,0)
#change value of V0[] for differnet initial value eg. V0[i]=20*abs(math.sin(math.pi*(lower_x+hx*i)))
#if a function distribution is given then set the values in the for loop for V0
#tot_steps is the total steos for second i loop
#For Q3 in Assignment 6 sample input is lib.PDE_sol(2,100,4,80000,0,5000)
#Change initial value of V0
#Nx is number of steps in x and Nt is number of steps in t
def PDE_sol(lx,Nx,lt,Nt,lower_x,tot_steps):
    step_arr = [10, 20, 50, 100, 200, 500, 1000, tot_steps]
    hx=(lx/Nx)
    ht=(lt/Nt)
    alpha=ht/(hx)**2
    V0=np.zeros(Nx+1)
    V1=np.zeros(Nx+1)
    x_cor = np.linspace(lower_x, lx, Nx + 1)
    ctr=0 #marker for the value in step_arr
    #if alpha<=0.5:print("Stability can be a problem")
    # Change initial value here
    for i in range(0,Nx+1):
        if lower_x + (hx * i) == 1:#1 as inital 300C at length 1
            V0[i]=300
        else:
            V0[i]=0
        x_cor[i]=(lower_x + hx * i)
    plot.plot(x_cor, V0, label=0)
    #Matrix mult for sparse when only some are multiplied
    for j in range(0,tot_steps+1):#1000 is number of steps taken
        for i in range(0,Nx+1):

            if i==0:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i+1]
            elif i==Nx:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i-1]
            else:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i-1]+alpha*V0[i+1]
        for k in range(0,Nx+1):#Equating array V0 to V1
           V0[k]=V1[k]
        if j==step_arr[ctr]:
           plot.plot(x_cor,V1,label=step_arr[ctr])
           #print(V0[50])
           ctr=ctr+1
    plot.legend()

#correct code for transpose, use this
def prop_transpose(x):
    r_c=list(np.shape(x))
    x_t=np.zeros([r_c[1],r_c[0]])
    for i in range(0,len(x)):
        for j in range(0,len(x[0])):
            x_t[j][i]=x[i][j]
    return x_t
#Midpoint method for numerical integration
#l1 and l2 are upper and lower limits and N is number of divisions
def midpoint_integration(fn,N,l1,l2):
    h=(l1-l2)/N
    sum=0
    for i in range(0,N):
       sum+=function_midpoint(fn,((l2+i*h)+(l2+(i+1)*h))/2)*h
    return sum
def function_midpoint(fn,x):
    return eval(fn)
#Simpson method for numerical integration
#l1 and l2 are upper and lower limits and N is number of divisions
# ub is error upper bound
#Here N is not an input, use other if N is input
def simpson_integration_finds_N(fn,l1,l2,ub,deri_max):
    N=find_N_simpson(ub,l2,l1,deri_max)
    print(N)
    h=abs(l1-l2)/(N)#h=(x2-x0)/2
    sum=0
    for i in range(0,N):
       x0=l2+h*i
       x2=l2+h*(i+1)
       x1 =(x0+x2)/2  # midpoint
       sum+=fn_simpson(fn,x0)+4*fn_simpson(fn,x1)+fn_simpson(fn,x2)
    return sum*h/6
def fn_simpson(fn,x):
    return eval(fn)
def find_N_simpson(ub,l2,l1,dm):
    return math.ceil(((l1-l2)**5/(180*ub)*dm)**(1/4))
#Simpson method for numerical integration
#l1 and l2 are upper and lower limits and N is number of divisions
#Here N is not an input, use other if N is input
def simpson_integration_N(fn,l1,l2,N):
    h=abs(l1-l2)/(N)#h=(x2-x0)/2
    sum=0
    for i in range(0,N):
       x0=l2+h*i
       x2=l2+h*(i+1)
       x1 =(x0+x2)/2  # midpoint
       sum+=fn_simpson_N(fn,x0)+4*fn_simpson_N(fn,x1)+fn_simpson_N(fn,x2)
    return sum*h/6
def fn_simpson_N(fn,x):
    return eval(fn)
#Trapezoid method for numerical integration
#l1 and l2 are upper and lower limits and N is number of divisions
def trapezoid_integration(fn,N,l1,l2):
    h=abs(l1-l2)/N
    sum=0
    for i in range(0,N):
       x0=l2+i*h
       x1=l2+(i+1)*h
       sum+=fn_trap(fn,x0)+fn_trap(fn,x1)
    return sum*h/2
def fn_trap(fn,x):
    return eval(fn)

#eventhough the convergence is slow, it is the only known method to calculate integration for multiple variables
#N is number of random numbers picked
#a and b [a,b] is domain of integration
#interval is increase in step of N
#start is starting value of N
#Check why we need to multiply with interval
#This one plots convergence according to number of steps
def monte_int_plot(N,fn,a,b,interval,start):
    sum=0
    sum2=0
    F=[]
    n=[]#to store values of N
    sig_2=[]
    for i in range(0,N+1,interval):#intial value is 10 and increases in value of 10
      X=a+(b-a)*rand_no_LCG_pos_k(10,i+1,1)
      sum+=fn_monte(X,fn)
      sum2+=fn_monte(X,fn)**2
      if (i)%interval==0:#change initial value of i here
        if i!=0:
         F.append((b - a) / i * sum*interval)
         sig_2.append(1/i*sum2-(1/i*sum)**2)
         n.append(i)
    plot.xlabel("Number of steps")
    plot.ylabel("Integration")
    plot.plot(n,F)
    plot.show()
    #print(F)
    #print(n[len(n)-1])
    return F[len(F)-1],sig_2[len(sig_2)-1]

def fn_monte_plot(x, fn):
    y = eval(fn)
    return y

#eventhough the convergence is slow, it is the only known method to calculate integration for multiple variables
#N is number of random numbers picked
#a and b [a,b] is domain of integration
#interval is increase in step of N
#start is starting value of N
#Check why we need to multiply with interval
#This and plotting monte carlo give slightly diff value becuase different random number are taken due to different starting seed
def monte_int(N,fn,a,b):
    sum=0
    sum2=0
    F=[]
    n=[]#to store values of N
    sig_2=[]
    for i in range(1,N+1):#intial value is 10 and increases in value of 10
      X=a+(b-a)*rand_no_LCG_pos_k(10,i,1)
      sum+=fn_monte(X,fn)
      sum2+=fn_monte(X,fn)**2
      F.append((b - a) / i * sum)
      sig_2.append(1/i*sum2-(1/i*sum)**2)
    #print (F)
    #print(len(F))
    return F[len(F)-1]


def fn_monte(x, fn):
    # print(x)
    y = eval(fn)
    return y
#To get eigenvalue and eigenvector
#WRONG REctify, in eigenvector the sign is opposite of what it should be
def e_val(A):
    #A=read_file_aug(filename)
    x0=np.zeros([len(A),1])
    for i in range(0,len(A)):
        x0[i][0]=(rand_no_LCG_pos_k(10,i,3))#taking scaling factor as i+2
    denom=list(matrix_mult(A,x0))#to store denominater
    denom_dot_1=matrix_mult(prop_transpose(denom),x0)#value after dotting with y i.e. x0 in this case
    lamb1=1
    lamb2=0
    k=1
    while k<=50:
        lamb1=lamb2
        denom=list(matrix_mult(A,denom))
        #print(denom)
        denom_dot_k=matrix_mult(prop_transpose(denom),x0)
        lamb2=denom_dot_k/denom_dot_1
        denom_dot_1=denom_dot_k
        k=k+1
        if abs(lamb1-lamb2)<0.001:
            break
    sum=0
    #Normalizing
    for i in range(0,len(denom)):
        sum+=denom[i]**2
    for i in range(0, len(denom)):
        denom[i]=denom[i]/(sum**0.5)
    return lamb2,denom,k# lamb2 is eigenvalues,denom is eigenvector and k is number of iterations taken

#transpose of the matrix for complex matrix x
def prop_transpose_complex(x):
    r_c = list(np.shape(x))
    x_t =[]
    y=[]
    for i in range(0, len(x[0])):
        for j in range(0, len(x)):
            y.append(x[j][i])
        x_t.append(y)
        y=[]
    return x_t
def prop_transpose_complex_1d(x):
    x_t =[]
    y=[]
    for i in range(0, len(x)):
            y.append(x[i])
            x_t.append(y)
            y=[]
    return x_t
#dot product of two matrix
# def dot_matrix(A,B):
#     r_c = list(np.shape(A))
#     D=np.zeros([r_c[1],r_c[0]])
#     for i in range(0, len(A)):
#         for j in range(0, len(A)):
#             D[i][j]=A[][]*B[][]

#Matrix multiplicaion of two complex matrix n and m
#returns -99999 if multiplication not possible
def mat_mult_complex(n,m):
    q = len(n)
    # checking if m is 1 d and treating accordingly for n
    if np.ndim(n) == 1:
        w= 1
    else:
        w = len(n[0])
    r = len(m)
    #checking if m is 1 d and treating accordingly for m
    if np.ndim(m) == 1:
        c=1
    else:
        c = len(m[0])
    if (w != r):
        print("Multiplication not possible")  # Condition for valid multiplication
        return -99999

    x = []
    y=[]
    sum = 0
    for k in range(0, q):

        for i in range(0, c):
            for j in range(0, r):
                if np.ndim(m)==1:
                    sum = sum + (n[k][j] * m[j])
                else:
                    sum = sum + (n[k][j] * m[j][i])  # Multiplication
            y.append(sum)
            sum = 0  # Reinitializing
        x.append(y)
        y=[]
    # print(x)
    return x
#returns the normaization factor for 1d or 2d column matrix
def norm(x):
    sum=0
    if np.ndim(x)==1:
        for i in range(0,len(x)):
           sum+=x[i]**2
        sum=sum**(0.5)
    else:
        for i in range(0, len(x[0])):
          for j in range(0, len(x)):
            sum += x[j][i] ** 2
        sum = sum ** (0.5)
    return sum

#Pritnting matrix
def print_matrix(x):
    for i in range(0,len(x)):
        for j in range(0,len(x[0])):
            print(x[i][j],"   ",end="")
        print()

#input matrix can have complex numbers also. Seperate consecutive elements with ",". VVVVI
#Use this for real inputs also
def read_file_complex(file_name):
    ctr = 0
    n=[]
    str_coeff=''#temporary string to read from file
    a=[]#temprory storing of numbers

    #Reading from file
    with open(file_name) as file:
        while True:

            content = file.readline()
            content = content.replace(" ", "")  # removing all spaces from the line for easier reading
            if not content:
                break
            sign = 1
            for str in content.split(","):
                 str=str.replace(" ","")
                 n.append(complex(str))
            a.append(n)  # making array
            n=[]
    return a

#rearranging s.t. first eigenvalue is greater than second and so forth
def rearrange(e_vec,e_val):
    max=-99999999
    for i in range(0,len(e_val)):
        ctr=-1
        for j in range(i,len(e_val)):
            if e_val[j]>e_val[i]:
                ctr=j
        if ctr!=-1:
           temp = e_val[i]
           e_val[i] = e_val[ctr]
           e_val[ctr] = temp
           e_vec=swap_column(i, ctr, e_vec)
    return e_vec,e_val

#swaps column c1 with c2 of array A
def swap_column(c1,c2,A):
    for i in range(0,len(A)):
        temp=A[i][c1]
        A[i][c1]=A[i][c2]
        A[i][c2]=temp
    return A

def nullspace(A):
    for i in range(0,len(A)):
         A[i].append(0)
    A=Gauss_Jordan_svd(A)
    print(A)
    A=[[10, -16, -4, 6], [-16, 32, 0,-16],[-4,0,8,4],[6,-16,4,10]]
    for i in range(0,len(A)):
         A[i].append(0)
    A = Gauss_Jordan_svd(A)
    b=find_null_vec(A)
    print(A)
#nullspace([[-1,3,2],[1,1,0],[1,1,0]])
# def find_null_vec(n):
#     for i in range(0,len(n)):
#         if n[i][i]==1:


#to write RK4 for question 1 of Assignment 6. The question is decomposed into the following equations:
# v=dy/dt
#Substituting
#2*dv/dt=math.cos(wt)-gamma v-2y
#Hence our equations to use are dy/dt=v and 2*dv/dt=math.cos(wt)-gamma v-2y
#put input function in dyt and dvt
#y0 and v0 are inital values and h is step length
#end is end value
#If doubt then check Assignment 6 Q1
#Sample input y,v,t = lib.RK4_1st_2nd_deriv(2, -1, 0.1, 100)
def RK4_1st_2nd_deriv(y0, v0, h, end):
    y_i = y0
    v_i = v0
    step = 0
    yl = [y0]
    vl = [v0]
    tl = [0]
    while step <= end:#implementing the algorithm to find the values at each step.
        k1y = h*dyt(y_i, v_i, step)
        k1v = h*dvt(y_i, v_i, step)

        k2y = h*dyt(y_i + k1y/2, v_i+k1v/2, step+h/2)
        k2v = h*dvt(y_i + k1y/2, v_i+ k1v/2, step+h/2)

        k3y = h*dyt(y_i+k2y/2, v_i+k2v/2, step+h/2)
        k3v = h*dvt(y_i+k2y/2, v_i+k2v/2, step+h/2)

        k4y = h*dyt(y_i+k3y, v_i+k3v, step+h)
        k4v = h*dvt(y_i+k3y, v_i+k3v, step+h)

        y_i += (k1y+2*k2y+2*k3y+k4y)/6
        v_i += (k1v+2*k2v+2*k3v+k4v)/6
        # print("ky =", k1y, k2y, k3y, k4y)
        # print("kv =", k1v, k2v, k3v, k4v)
        # print("h =", h)
        yl.append(y_i)
        vl.append(v_i)
        step += h
        tl.append(step)

    return yl, vl, tl
def dyt(y, v, t):
    f = v
    return f

def dvt(y, v, t):
    f = -0.02*v-10
    return f

#to solve the boundary value problem using shooting method.
#z0 =-0.5, -2
#decompose eq into
#dy/dx = z
#dz/dx = -a(T_a - T)
#start value is taken as 0, change h = (upper-0)/N if needed
#interpol determines if i want graph or value for interpolation
#end is end value
#If doubt then check Assignment 6 Q2
#z0 is inital guess
def RK4_boundary_value(y0, z0,x0, N, end, interpol):
    y_i = y0
    z_i = z0
    step = 0
    yl = [y0]
    zl = [z0]
    xl = [x0]
    h = (end-0)/N
    while step <= end:
        k1y = h*boundary_RK4_dyx(y_i, z_i, step)
        k1z = h*boundary_RK4_dzx(y_i, z_i, step)

        k2y = h*boundary_RK4_dyx(y_i + k1y/2, z_i+k1z/2, step+h/2)
        k2z = h*boundary_RK4_dzx(y_i + k1y/2, z_i+k1z/2, step+h/2)

        k3y = h*boundary_RK4_dyx(y_i+k2y/2, z_i+k2z/2, step+h/2)
        k3z = h*boundary_RK4_dzx(y_i+k2y/2, z_i+k2z/2, step+h/2)

        k4y = h*boundary_RK4_dyx(y_i+k3y, z_i+k3z, step+h)
        k4z = h*boundary_RK4_dzx(y_i+k3y, z_i+k3z, step+h)

        y_i += (k1y+2*k2y+2*k3y+k4y)/6
        z_i += (k1z+2*k2z+2*k3z+k4z)/6
        yl.append(y_i)
        zl.append(z_i)
        step += h
        xl.append(step)

    if interpol == 0:
        return zl, yl, xl
    else:
        return z_i, y_i

def boundary_RK4_dyx(y,z,x):
    f = z
    return f

def boundary_RK4_dzx(y,z,x):
    f = 0.01*(y-20)
    return f

# Q2 Assignment 6
#Shooting star method with RK4 integrator
#Sample input x,y=lib.boundary_value_corrected()
#Here dy/dx=z, in Q2 assignment 4 z=dT/dx
#Hence if you want to plot dy/dx thhen use z and x
#Change eq in RK4 integrator i.e. RK4_boundary_value(y0, z0,x0, N, end, interpol)
def boundary_value_corrected():
   y_L = 200#Temp /value of y at length L
   y_0=40#Temp/value of y at 0
   N = 1000#Iterations
   z1 = -0.5#inital guess
   z1h, y1 = RK4_boundary_value(y_0, z1,0, 100, 10, 1)#initialiing
   z2 = -2#inital guess
   z2h, y2 = RK4_boundary_value(y_0, z2,0, 100, 10, 1)#initialiing
   iter = 0
   while abs(y1-y_L) >= 0.001 and abs(y2-y_L)>=0.001 and iter <= 30:
      iter +=1
      znew = z2 + ((z1 - z2)*(y_L - y2))/(y1 - y2)
      znew2, ynew = RK4_boundary_value(y_0, znew,0, N, 10, 1)
      #print(ynew, znew)
      if abs(ynew - y_L) < 0.001:
          z, y, x = RK4_boundary_value(y_0, znew,0, N, 10, 0)
          break
      else:
          if ynew < y_L:
              z2 = znew
              y2 = ynew
          else:
              z1 = znew
              y1 = ynew
      if iter==30:
          print("Convergence not obtained in 30 iterations")
   #print(iter)
   #print(y)
   return x,y,z

#Radioactive Decay
#A deacys to B and B to C
#N_A and N_B are initial number of A and B. taking N_B as 0 initially
#dt is smallest time interval say 0.0001
#tao_A and tao_B are half life respectively of A and B
#lamb denotes lambda i.e. proportionality constant lamb_A for A and lamb_B for B
#T is time to which we observe
def radioactive_decay(N_A,N_B,T,dt,tao_A,tao_B):
    lamb_A=np.log(2)/tao_A
    lamb_B = np.log(2) / tao_B
    ctr=0
    N_C=0#taking initial number of particle C to be 0
    N_A_arr=[]
    N_B_arr = []
    N_C_arr = []
    T_arr=np.arange(0,T,dt)
    for t in T_arr:
        #Deacy of A
        for i_a in range(0,N_A):
            if rand_no_LCG_pos_k(10,ctr,1)<=lamb_A*dt:
                N_A=N_A-1
                N_B=N_B+1
            ctr += 1
        #Deacy of B
        for i_b in range(0,N_B):
            if rand_no_LCG_pos_k(10,ctr,1)<=lamb_B*dt:#Maybe missing a term here check from slides TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
              if N_B>=1:
                N_B=N_B-1
                N_C=N_C+1
            ctr += 1
        N_A_arr.append(N_A)
        N_B_arr.append(N_B)
        N_C_arr.append(N_C)
    plot.scatter(T_arr,N_A_arr)
    plot.scatter(T_arr,N_B_arr)
    plot.scatter(T_arr,N_C_arr)
    plot.show()

#left right box
def box(N,T,dt):
    nl=N#number of particle in the left, itially eqaul to N
    ctr=0
    N_left_t=[]
    T_arr=np.arange(0,T,dt)
    ctr2=0
    for t in T_arr:
        # Now for transition from left
            if rand_no_LCG_pos_k(10,ctr,1)<=nl/N*dt:
                nl=nl-1
                ctr += 1
                ctr2+=1
            #Now for transition from right
            elif rand_no_LCG_pos_k(10,ctr,1)<=(1-nl/N)*dt:#Maybe missing a term here check from slides TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
               nl=nl+1
               ctr += 1
            else:
                ctr=ctr+1
            N_left_t.append(nl)
            #print(nl)
            # print(ctr2)
            # print(1-nl/N)
            # print(rand_no_LCG_pos_k(10,ctr,1))
    plot.scatter(T_arr,N_left_t)
    plot.show()

def box2(N,T,dt):
    nl=N#number of particle in the left, itially eqaul to N
    ctr=0
    N_left_t=[]
    N_right_t = []
    T_arr=np.arange(0,T,dt)
    ctr2=0
    prev=rand_no_prev(100,1)
    for t in T_arr:
        # Now for transition from left
            prev=rand_no_prev(prev,1)
            if prev<=nl/N:
                nl=nl-1
            #Now for transition from right

            elif prev<=(1-nl/N):
               nl=nl+1
            else:
               prev=rand_no_prev(prev,1)
            #print(prev)
            N_left_t.append(nl)
            N_right_t.append(5000-nl)
    plot.xlabel("Time")
    plot.scatter(T_arr,N_left_t,label="No. in Left half")
    plot.scatter(T_arr,N_right_t,label="No. in Right half")
    plot.legend()
    plot.show()

def plot_polynomial(x):
    y=[]
    for i in range(0,len(x)):
        y.append(0.2546-1.1937*x[i][0]-0.4572*(x[i][0]**2)-0.80256*(x[i][0]**3)+0.0132*(x[i][0]**4))
    return y

#Now for inverse using LU

#LU decomposition with forward backward
def LUdecomp_inv(a):
    # Array on RHS is simply identity matrix
    b=np.zeros((len(a),len(a[0])))
    for i in range(0,len(a)):
        for j in range(0,len(a[0])):
            if i==j:
                b[i][j]=1
            else:
                b[i][j]=0
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