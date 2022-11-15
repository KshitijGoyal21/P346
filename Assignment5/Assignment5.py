import library as lib
def Q1(fn,l1,l2):
    print("               Midpoint              Trapezoid              Simpson")
    print("N=10","   ",lib.midpoint_integration(fn,10,l1,l2),"    ",lib.trapezoid_integration(fn,10,l1,l2),"    ",lib.simpson_integration_N(fn,l1,l2,10))
    print("N=20", "   ", lib.midpoint_integration(fn, 20, l1, l2), "    ", lib.trapezoid_integration(fn, 20, l1, l2),"     ", lib.simpson_integration_N(fn, l1, l2, 20))
    print("N=30", "   ", lib.midpoint_integration(fn, 30, l1, l2), "      ", lib.trapezoid_integration(fn, 30, l1, l2),"    ", lib.simpson_integration_N(fn, l1, l2, 30))
    print("We notice that Simpson is the closest to actual value")
fn="math.sqrt(1+1/x)"
l1=4
l2=1
Q1(fn,l1,l2)
#OUTPUT
#                Midpoint              Trapezoid              Simpson
# N=10     3.6189788939398126      3.6226083803599547      3.620188722746527
# N=20     3.6198800323016473      3.620793637149885       3.62018456725106
# N=30     3.62004881243588        3.6204553882819273      3.620184337717895
#We notice that Simpson is the closest to actual value


def Q2(fn,l1,l2):
    result,sig_2=lib.monte_int_plot(2000, fn, l2, l1, 10,10)
    print("Integration using Monte Carlo",result,"\u00B1",sig_2)
    #print(lib.monte_int(2000,fn,l2,l1))
fn="math.sin(x)*math.sin(x)"
l1=1
l2=-1
Q2(fn,l1,l2)
#OUTPUT
#Integration using Monte Carlo 0.5366663637437515 ± 0.011038166131656762
#Graph of convergence is also given


def Q3():
    l1=2
    l2=0
    #We form the function for getting centre of mass
    #Centre of mass is given by integration from 0 to 2 of (Mass density * position)/Total Mass
    numerator=lib.simpson_integration_N("x**2*x",2,0,1000)#integration from 0 to 2 of (Mass density * position)
    denominator=lib.simpson_integration_N("x**2",2,0,1000)#Total mass
    print("Centre of Mass is at length,",numerator/denominator)
Q3()
#OUTPUT
#Centre of Mass is at length 1.500000000000001