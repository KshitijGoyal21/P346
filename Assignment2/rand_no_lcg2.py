# importing the required module
import matplotlib.pyplot as plot
import numpy as npy
import math

def rand_no(iter,x):#x is the seed
    a = 1103515245
    c = 12345
    m = 32768
    y=[]
    for i in range(0,iter):
       y.append((((a*x+c)%m)/m))
       x=((a*x+c)%m)
    return y
rand_no(300,10)