import numpy as np
"""
class cl:
    def tes():
        print(1)
        
    def tes2(**kw):
        x=getattr(cl,"tes")
        x()
        for i in kw:
            print(i)
        d=("rf",'kmer')
        print(d[0],d[1])


def main(**kw):
    x=cl()
    a="tes2"
    c="cl"
    b=getattr(cl,a)
    b(**kw)


def bianli():
    models=[]
    for i in [1,2,4,5]:
        x=str(i)
        models.append(i)
    print(models)
"""

#main(xb=213)
#bianli()
"""
x=135671
b=0
for i in range(5):
    b=b+int(x*0.2)
    print(b)
"""
#print(1>=0.5)
"""
lis1=[[1,3],[2,4],[3,6]]
lis2=[[3],[4],[5]]
x=np.append(lis1,lis2,axis=1)
print(x)


"""
f=open("fullDataset.csv",'r')
lines = f.readlines()
# 使用len()函数计算列表长度，即为文件行数
num_lines = len(lines)
print(num_lines)
