import library as lib

def Q1(fn):
    #no root in the given interval
    print("Using bracketing")
    print("For bisection")
    a = lib.fn_bisection(1.5, 2.5,fn)  # The input is the interval and function
    print("For regula falsi")
    b = lib.root_reg_fal(1.5, 2.5,fn)  # The input is the interval and function
    print("Output from Bisection method", a)
    print("Output from Regula Falsi method", b)
Q1("(math.log(x/2)-math.sin(5*x/2))")
#OUTPUT
#Using bracketing
# Output from Bisection method 1.4019301414489747
# Output from Regula Falsi method 1.4019299353346426
#ctr             Bisection                       Regula Falsi
# count= 1       1.425                        1.4089960356059543
# count= 2       1.3875000000000002           1.4023285745972491
# count= 3       1.40625                      1.4019519838618306
# count= 4       1.396875                     1.4019311501433975
# count= 5       1.4015625                    1.4019299989420133
# count= 6       1.40390625                   1.4019299353346426
# count= 7       1.402734375
# count= 8       1.4021484375000002
# count= 9       1.40185546875
# count= 10       1.402001953125
# count= 11       1.4019287109375
# count= 12       1.4019653320312502
# count= 13       1.401947021484375
# count= 14       1.4019378662109374
# count= 15       1.4019332885742188
# count= 16       1.4019309997558596
# count= 17       1.4019298553466797
# count= 18       1.4019304275512696
# count= 19       1.4019301414489747
# count= 20       1.401929998397827



def Q2(fn2,fn2_der):

    a=lib.fn_bisection(-1,0,fn2)#The input is the bracketing variables
    b=lib.root_reg_fal(-1,0,fn2)
    c=lib.newton_rhapson_root(fn2,fn2_der)
    print("Output from Bisection method",a)
    print("Output from Regula Falsi method", b)
    print("Output from Newton Rhapson method", c)
    print("Actual root is -0.73908513321516")
    print("Hence Newton Rhapson is most accurate,correct to 12 digits")
Q2("-x-math.cos(x)","math.sin(x)-1")
#OUTPUT
#Output from Bisection method -0.7391357421875
#Output from Regula Falsi method -0.7390851156443783
#Output from Newton Rhapson method -0.7390851332159002
#Actual root is -0.73908513321516
#Hence Newton Rhapson is most accurate,correct to 12 digits

def Q3():
    a =lib.Laguerre_poly_root([4,0,-5,0,1])
    print(a)
Q3()
#OUTPUT
# -2.000000302898602
# -1.0000003028986022
# 1.0000021202937015
# 1.999998485503503

def Q4():
    #Reading from file
    x,y=lib.read_file("assign4fit.txt")
    a=lib.poly_fit(x,y,3)
    for i in range(0,len(a)):
        print(a[i][len(a)],"x^",i,)

Q4()
# OUTPUT
# 0.5746586674196181 x^ 0
# 4.725861442141881 x^ 1
# -11.128217777643187 x^ 2
# 7.66867762290941 x^ 3
