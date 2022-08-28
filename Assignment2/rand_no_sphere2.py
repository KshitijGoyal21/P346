# importing the required module
import rand_no_lcg2 as rg
def sphere(iter,seed):#iter is the number of iterations
    ctr=0
    #Taking 3 list of random numbers
    x= rg.rand_no(iter,seed)
    y= rg.rand_no(iter,seed+20)
    z= rg.rand_no(iter,seed+21)
    for i in range(0,iter):
         #x.append(i/iter)
         if ((x[i])**2+(y[i])**2+(z[i])**2<=1):#Checking if inside octent
                 ctr=ctr+1
    print((ctr)/iter)#Calculating volume
with open("sphere_input.txt") as a:#reading input from file
 contents = a.readlines()
 for i in range(0,len(contents),2):
    sphere(int(contents[i]),int(contents[i+1]))


"""
Output
0.52358
"""