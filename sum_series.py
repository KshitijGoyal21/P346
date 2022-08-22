import matplotlib.pyplot as plot
def sum_series():
    sum=0
    a=[]
    n=list(range(1,14))
    for i in range(1,14):
         sum=sum+((-1)**(i+1))/(2**i)
         a.append(sum)
    print("Sum is ",sum)
    plot.plot(n,a)
    plot.show() #showing plot
sum_series()