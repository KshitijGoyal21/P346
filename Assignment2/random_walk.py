import rand_no_lcg2 as rg
import matplotlib.pyplot as plot
def random_walk(iter,seed):#iter is the number of iterations

    cor1= rg.rand_no(iter,seed)#taking random numbers in an array for x coordinates
    cor2= rg.rand_no(iter,seed+10)#taking second array of random numbers for y coordinates
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
    print("rms value is",(rms/(iter))**0.5)#Calculating rms
    print("Net displacement is",(sum1**2+sum2**2)**(0.5))#Calculating net displacement
    plot.plot(x,y)
    plot.show() #showing plot
with open("random_walk_input.txt") as a:#reading from file
 contents = a.readlines()
for i in range(0,len(contents),2):
    random_walk(int(contents[i]),int(contents[i+1]))


"""
Output
For 300 steps
rms value is 0.8214449204069914
Net displacement is 6.454176617631995

For 600 steps
rms value is 0.8359173652490888
Net displacement is 22.95735601721538

For 900 steps
rms value is 0.8129668657971751
Net displacement is 22.6860226988925
"""