import library as lib
import numpy as np
from numpy.linalg import eigh
import math
from numpy.linalg import svd
import inspect
from scipy.linalg import svd
#Does not work for singualr matrix because python does not return eigenvalue and eigenvector i.e. for singular value 0
def main(A):
   # source_DF = inspect.getfile(svd)
   # print(source_DF)
   # b=[]
   # svd(b)
   A_t=lib.prop_transpose_complex(A)
   print(A)
   A_A_t=lib.mat_mult_complex(A_t,A)
   print(A_A_t)
   e_val,e_vec=eigh(A_A_t)#eigenvector is e_vec[0:i] where i is 1...n where n is dimension of A_A_t. i.e. columns represent eigenvectors
   print(e_val)
   print(e_vec)
   #rearranging s.t. first eigenvalue is greater than second and so forth
   e_vec,e_val=lib.rearrange(e_vec,e_val)
   print(e_vec)
   print(e_val)
   W = np.zeros([len(A), len(A[0])])
   print(W)
   for i in range(0, len(A[0])):
      if e_val[i] > 0:
         W[i][i] = math.sqrt(e_val[i])
   U=[]
   #Normalizing eigenvectors
   for i in range(0,len(e_vec)):#columns
     A_e_vec =lib.mat_mult_complex(A,e_vec[:,i])
     norm_factor=lib.norm(A_e_vec)#For A multiplied by eigenvector
     print(A_e_vec,"dfsjcsjk",e_vec[:,i])
     print(norm_factor,"sdcmdsc")
     if norm_factor!=0:#maybe it has to be eigenvalue !=0  jkfvnskjkjvdfvkjdk
        U.append(A_e_vec/norm_factor)
   #print(U)
   U=lib.prop_transpose_complex(U)
   print("U")
   lib.print_matrix(U)
   print("W")
   lib.print_matrix(W)
   print("V^T")
   lib.print_matrix(lib.prop_transpose_complex(e_vec))
   print("A on Multiplyting UWV^T")
   lib.print_matrix(lib.mat_mult_complex(U, lib.mat_mult_complex(W, lib.prop_transpose_complex(e_vec))))
   print("UU^T")
   lib.print_matrix(lib.mat_mult_complex(lib.prop_transpose_complex(U),U))
   print("VV^T")
   lib.print_matrix(lib.mat_mult_complex(e_vec, lib.prop_transpose_complex(e_vec)))
   print(svd(A))
#A=[[1,2,1],[4,5,3],[7,8,0]]
A=lib.read_file_complex("DIY_input_SVD.txt")
main(A)
#OUTPUT
# U
# [-0.18032145]    [0.28267262]    [0.94211484]
# [-0.53144239]    [0.77797694]    [-0.33514306]
# [-0.82767938]    [-0.56111325]    [0.00993828]
# W
# 12.715062097603413    0.0               0.0
#      0.0    2.6705922581813124          0.0
#      0.0              0.0       0.44173843463200413
# V^T
# -0.6370276938551417    -0.7581001051473142    -0.1395705837102894
# -0.1996607179216847    -0.012609958647787994    0.9797839489713527
# -0.7445342940258675    0.6520162724338493    -0.14332991837320835
# A on Multiplyting UWV^T
# [1.]    [2.]    [1.]
# [4.]    [5.]    [3.]
# [7.]    [8.]    [2.60425362e-16]
# UU^T
# [1.]              [5.55111512e-17]    [5.19029264e-15]
# [5.55111512e-17]      [1.]           [-5.39845946e-15]
# [5.19029264e-15]  [-5.39845946e-15]       [1.]
# VV^T
# 0.9999999999999996    4.440892098500626e-16    1.3877787807814457e-17
# 4.440892098500626e-16    0.9999999999999993    5.551115123125783e-17
# 1.3877787807814457e-17    5.551115123125783e-17    0.9999999999999998