import sys
import numpy as np
def mat_mult(n,m):
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
    print(x)
n=[[2,-3,1.4],[2.5,1,-2],[-0.8,0,3.1]]
m=[[0,-1,1],[1.5,0.5,-2],[3,0,-2]]
mat_mult(n,m)
n2=[[0,-1,1],[1.5,0.5,-2],[3,0,-2]]
m2=[[-2],[0.5],[1.5]]
mat_mult(n2,m2)
n1=[[1,0,-1]]
m1=[[-2],[0.5],[1.5]]
mat_mult(n1,m1)
#