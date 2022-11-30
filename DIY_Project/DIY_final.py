import library as lib
import numpy as np
import math
import matplotlib.pyplot as plot
from numpy.linalg import svd
from PIL import Image
#The input svd_limit is the number of singular values which we use for image reconstruction
def image_compression_main(originalImage,svd_limit,colour):
  factor=1
  PSNR=0
  if colour=="RGB":
    factor=3
    #First we seperate RGB colours in 3 array, 1 for each colour
    img = Image.open(originalImage).convert('RGB')
    red, green, blue,original=openimage(originalImage)
    #compressing using svd
    red_compress=svd_image_compress(red,svd_limit)
    green_compress = svd_image_compress(green, svd_limit)
    blue_compress=svd_image_compress(blue,svd_limit)
    #now we need to reconstruct image from our compressed array
    img_red=Image.fromarray(red_compress, mode=None)
    img_green = Image.fromarray(green_compress, mode=None)
    img_blue = Image.fromarray(blue_compress, mode=None)
    #merging all colours to form resultant image
    Image_compress=Image.merge('RGB',(img_red,img_green,img_blue))
    original_img_arr=np.asarray(original)
    Image_compress_arr=np.asarray(Image_compress)
    original.show()
    Image_compress.show()
    mr=np.shape(img_red)[0]
    mc=np.shape(img_red)[1]
    PSNR=PSNR_cal(original_img_arr,Image_compress_arr,mr,mc,"RGB")
  elif colour=="grey":
    img=Image.open(originalImage).convert('L')
    grey_compress=svd_image_compress(img,svd_limit)
    Image_compress= Image.fromarray(grey_compress)
    mr = np.shape(Image_compress)[0]
    mc = np.shape(Image_compress)[1]
    img.show()
    Image_compress.show()
    original_img_arr = np.asarray(img)
    PSNR=PSNR_cal(original_img_arr,grey_compress,mr,mc,"grey")
  originalsize=mr*mc*factor
  compress_Size=svd_limit*(1+mr+mc)*factor
  print("Original size",originalsize)
  print("Compress size",compress_Size)
  print("Percentage of elements in compressed",compress_Size/originalsize*100)
  #Image_compress.save("C:\\Users\\Kshitij Goyal\\PycharmProjects\\CompLab\\horse200.jpg")
  return PSNR

def PSNR_cal(original,Image_compress,mr,mc,colour):
    # To calculate PSNR
    MSE = np.longdouble(0)
    for i in range(0, len(original)):
        for j in range(0, len(original[0])):
            #print(original[i][j],Image_compress[i][j])
            MSE =np.longdouble( MSE+(original[i][j] - Image_compress[i][j])**2)
            #print(MSE)
    if colour=="RGB":
        mse=np.longdouble(MSE[0]+MSE[1]+MSE[2])/3
    else:
        mse=np.longdouble(MSE)
        print(MSE)
    mse =mse/(mr * mc)
    #print(mse)
    PSNR = 10 * math.log(((255**2)/mse),10)#in dB
    return PSNR

def openimage(image_path):
    original_img=Image.open(image_path)
    orig_img_arr=np.array(original_img)
    red=orig_img_arr[:,:,0]
    green = orig_img_arr[:, :, 1]
    blue = orig_img_arr[:, :, 2]
    return [red,green,blue,original_img]

#Compresses arr_input using SVD for specific SVD limit given by k
def svd_image_compress(arr_input,k):
    U,sig,V_T=svd(arr_input)
    #Multiplying compressed version of U with compressed version of sig
    compress_U_sig = np.matmul(U[:, 0:k], np.diag(sig)[0:k,0:k])
    arr_compress=np.array(np.matmul(compress_U_sig,V_T[0:k,:]))
    #To make int array
    arr_compress=arr_compress.astype('uint8')
    return arr_compress

#To create PSNR vs k graph
#iter denotes end point of iteration
def PSNR_graph(filename,colour,iter):
    x = []
    y = []
    for i in range(1, iter+2,10):
        print(i)
        y.append(image_compression_main(filename,i,colour))
        x.append(i)
    plot.scatter(x,y,color='black')
    plot.plot(x,y)
    plot.xlabel("SVD limit k")
    plot.ylabel("PSNR in dB")
    plot.title("Greyscale image")
    plot.show()



#image_compression_main('horse',500,"RGB")
#PSNR_graph('eagle',"grey",440)
#image_compression_main('horse',200)
#image_compression_main('horse',500)
#image_compression_main('eagle',50)
#image_compression_main('eagle',100)
#image_compression_main('eagle',150)


