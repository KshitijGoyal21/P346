#Question 1 Gauss Jordan
def GJ_matrix():
    ctr = 0
    n=[]
    str_coeff=''#temporary string to read from file
    a=[]#temprory storing of numbers
    #Reading from file
    with open("GJ_assignment3.txt") as file:
        while True:

            content = file.readline()
            content=content.replace(" ", "")#removing all spaces from the line for easier reading
            if not content:
                break
            for i in range(0, len(content), 1):
                t=ord(content[i])#ord funtions takes character and returns its ascii value
                if ord(content[i - 1]) == 45:
                    sign = -1
                else:
                    sign = 1
                if t>= 48 and t<= 57:
                    if i<len(content)-1:
                       if ord(content[i+1])>= 48 and ord(content[i+1])<= 57:
                           str_coeff = str_coeff + content[i]
                           continue
                       str_coeff = str_coeff + content[i]
                    a.append(sign * int(str_coeff))

                str_coeff = ""
            n.append(a)
            a=[]
            ctr = ctr + 1
    print("Input augumented matrix")
    print(n)
    #Gausss Jordan
    for i in range(0,len(n)):
        if n[i][i] == 0:
            k = max(n, i)
            n = swap(n, i, k)
        n = multiply(n, i, 1/n[i][i])
        #print(n)
        for j in range(0, len(n)):
            if n[j][i] != 0 and i != j:
                n = add_multi(n, j, -n[j][i], i)
                #print(n)
    print("Output")
    for i in range(0,len(n)):
        print(n[i][len(n)])


def max(n, i):#getting max element in array
    max = 0
    k = -1
    for j in range(i,len(n)):
        if max < abs(n[j][i]):
            max = abs(n[j][i])
            k = j
    return k


def swap(n, i, k):#swapping in array
    for j in range(0,len(n)+1):
        temp = n[i][j]
        n[i][j] = n[k][j]
        n[k][j] = temp
    return n


def multiply(n, i, m):
    for j in range(0, len(n)+1):
        n[i][j] = n[i][j] * m
    return n


def add_multi(n, j, c, i):
    for k in range(0, len(n)+1):
        n[j][k] = n[j][k] + c * n[i][k]
    return n


GJ_matrix()
#OUTPUT
# Input augumented matrix
# [[1, -1, 4, 0, 2, 9, 19], [0, 5, -2, 7, 8, 4, 2], [1, 0, 5, 7, 3, -2, 13], [6, -1, 2, 3, 0, 8, -7], [-4, 2, 0, 5, -5, 3, -9], [0, 7, -1, 5, 4, -2, 2]]
# Output
# -1.761817043997858
# 0.8962280338740127
# 4.051931404116157
# -1.6171308025395434
# 2.0419135385019147
# 0.15183248715593478
