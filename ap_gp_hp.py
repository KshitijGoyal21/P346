
def ap_gp_hp(n,a):
    sum_ap=0
    ap=a
    sum_gp=0
    gp=a
    sum_hp=0
    hp=a
    for i in range(1,n+1):# calculating sum
         sum_ap=sum_ap+ap
         ap=ap+1.5
         sum_gp=sum_gp+gp
         gp=gp*0.5
         sum_hp=sum_hp+hp
         hp=1/(a+1.5*i)
    print("Sum of ap ",sum_ap)
    print("Sum of gp ", sum_gp)
    print("Sum of hp ", sum_hp)
ap_gp_hp(5,1)