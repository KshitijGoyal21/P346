class MyComplex:
 def Complex():
    #Taking input
    a_r=int(input("Enter real part of 1st input"))
    a_i=int(input("Enter imaginary part of 1st input"))
    b_r=int(input("Enter real part of 2nd input"))
    b_i=int(input("Enter imaginary part of 2nd input"))
    print("Sum is",a_r+b_r,"+",a_i+b_i,"i")            #printing sum
    c_r=a_r*b_r-1*(a_i*b_i)
    c_i=a_r*b_i+b_r*a_i
    print("Product is",c_r,"+",c_i,"i")                #printing product
    print("Modulus of 1st input is",(a_r**2+a_i**2)**.5)     #printing modulus
    print("Modulus of 2nd input is",(b_r**2+b_i**2)**.5)     #printing modulus
 Complex()