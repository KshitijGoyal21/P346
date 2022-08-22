

def fact_sum(n):
    sum=0
    fact=1
    for i in range(1,n+1):
        if i%2!=0:
         sum=sum+i
        fact=fact*i
    print("Sum is ",sum)
    print("Factorial is ",fact)
fact_sum(5)