
import numpy as np
import math
import matplotlib.pyplot as plot
#Gauss jordan with augumented matrix
def Gauss_Jordan(n):
    #Gausss Jordan
    for i in range(0,len(n)):
        if n[i][i] == 0:
            k = max(n, i)
            n = swap(n, i, k)
        n = multiply(n, i, 1/n[i][i])
        #print(n)
        for j in range(0, len(n)):
            if n[j][i] != 0 and i != j:
                n = add_multi(n, j, -n[j][i], i)
                #print(n)
    print("Output")
    print(n)

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
def swap(n, i, k):
    for j in range(0,len(n)+1):
        temp = n[i][j]
        n[i][j] = n[k][j]
        n[k][j] = temp
    return n

#used in gauss jordan
def multiply(n, i, m):
    for j in range(0, len(n)+1):
        n[i][j] = n[i][j] * m
    return n

#used in gauss jordan
def add_multi(n, j, c, i):
    for k in range(0, len(n)+1):
        n[j][k] = n[j][k] + c * n[i][k]
    return n
#General file read with sign and decimal but returns array a as 2d and b as 1d
def read_file(file_name):
    ctr = 0
    n = []
    str_coeff = ''  # used to take input from file
    x = []  # temprory storing of numbers
    y = []
    sign=1
    # Reading from file
    with open(file_name) as file:
        while True:

            content = file.readline()
            content = content.replace(" ", ",")  # removing all spaces from the line for easier reading
            if not content:
                break
            sign=1
            for i in range(0, len(content), 1):
                t = ord(content[i])  # ord funtions takes character and returns its ascii value
                if ord(content[i - 1]) == 45:
                    sign = -1  # deciding sign
                elif ord(content[i - 1]) == 43:
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
#Guass seidel with positive definite and is_symmetric with accuracy 10^-6 and diagonal dominance
def gs(a,b):
    x = np.zeros(len(a))
    e = list(x)  # matrix to stor error for each element
    f = 0  # to check if all elements have epsilon in the required bound(tolerence)
    ctr = 0
    sum = 0
    sum2 = 0
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
    print("Matrix for LHS", b)
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
    print("Number of iterations",ctr)


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
    for i in range(0, len(a)):
        if abs(a[i][i])<sum_row(a[i],a[i][i]):
            for j in range(0,len(a)):
                if i!=j:
                    if abs(a[i][j])>=sum_row(a[i],a[i][j]):
                       a=swap(a,i,j)
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
#swap function for 2d array
def swap(a,i,j):
    for k in range(0,len(a)):
        temp=a[i][k]
        a[i][k]=a[j][k]
        a[j][k]=temp
    return a
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
def interpol(x,y,a,n):
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
#It can give us the root of any function eg.x-2cosx=0 etc just write the function in fn method
def newton_rhapson_root():
    e = 0.0001  # epsilon
    d=0.0001  #delta
    c_prev=-9999#to store previous
    c=1#initial guess
    ctr=0
    while abs(c-c_prev)>e or fn(c)>d:
         c_prev = c
         ctr=ctr+1
         c=c-(fn(c)/fn_derivative(c))
    print(ctr)
    print(c)
    print(fn(c))
#write the function which is the input here
def fn(x):
    return (x*math.exp(x)-2)
#write the derivative of function which is the input here
def fn_derivative(x):
    return (math.exp(x)+x*math.exp(x))

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
    Gauss_Jordan(cal_x)
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
#LCG random number, use this in program
#gives the kth random number in the sequence of a perticular seed x
def rand_no_LCG_pos_k(x,k):
    # x axis values
    a = 1103515245
    c = 12345
    m = 32768
    # corresponding y axis values
    for i in range(0,k):
       x=((a*x+c)%m)
    return x/m
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
#bracketing included
def root_reg_fal(a, b):
    e = 0.000001  # epsilon
    d=0.000001  #delta
    while fn_reg_fal(a) * fn_reg_fal(b) > 0:#check bracketing
        if abs(fn_reg_fal(a)) > abs(fn_reg_fal(b)):

            b = 1.5 *(b - a)
        else:
            a =1.5 *(b - a)
    #print(a,"  ",b)
    c_prev=0
    c=-999
    ctr=0
    while abs(c-c_prev)>e or fn_reg_fal(c)>d:
         #print(a,"   ",b)
         c_prev = c
         ctr=ctr+1
         if fn_reg_fal(a)*fn_reg_fal(b)<0:
             c=b-((b-a)*fn_reg_fal(b))/(fn_reg_fal(b)-fn_reg_fal(a))
             if fn_reg_fal(a)*fn_reg_fal(c)<0:
                 b=c
             else:
                 a=c
         else:
             if abs(fn_reg_fal(a))>abs(fn_reg_fal(b)):
                  b=b-1.5*(b-a)
             else:
                  a=a+1.5*(b-a)
         #print(c,"psp")
    #print(ctr)
    return(c)#root
    #print(fn_reg_fal(c))#has to be close to zero
#input function to regula falsi
def fn_reg_fal(x):
    return (x**4-4*x**3-x**2+10*x)

#bisection to calucate root of polynomial in the range of x between a and b, update e and d to change accuracy
#change e and d to increse accuracy
#bracketing included
def fn_bisection(a, b):
    e = 0.0001  # epsilon for x value accuracy
    d=0.0001  #delta for y value accuracy
    while fn2(a) * fn2(b) > 0:#check bracketing
        if abs(fn2(a)) > abs(fn2(b)):

            b = 1.5 *(b - a)
        else:
            a =1.5 *(b - a)
    #print(a,"  ",b)
    ctr=0
    while abs(b-a)>e or fn2(abs(a))>d:
         ctr=ctr+1
         #print(a,"   ",b)
         if fn2(a)*fn2(b)<0:
             c=(a+b)/2
             if fn2(a)*fn2(c)<0:
                 b=c
             else:
                 a=c
         else:
             if abs(fn2(a))>abs(fn2(b)):
                  b=b-1.5*(b-a)
             else:
                  a=a+1.5*(b-a)
    return a
    #print(a) #value of root
    #print(fn2(a))#value of y at root
    #print(abs(b-a))# final difference between a and b
    #print(ctr) #interations
#input function to bisection
def fn2(x):
    return (x*math.exp(x)-2)
fn_bisection(1,4)