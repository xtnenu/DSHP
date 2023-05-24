from utils import loadcsv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def average(matrix):
    new_matrix = np.array(matrix).astype(float)
    col_means = np.mean(new_matrix, axis=0)
    return col_means

def three_baseline(path):
    x=loadcsv(path,' ')
    svm = x[::2]
    rf = x[1::2]
    GDBT = x[2::2]
    return average(svm),average(rf),average(GDBT)

def single(path)


data1=three_baseline("Data/pca-50.csv")
data2=three_baseline("Data/pca-200.csv")
data3=three_baseline("Data/pca-2000.csv")
data=data1+data2+data3
ax=sns.heatmap(data, cmap='coolwarm',annot=True, fmt='.3f')
ax.set_xlabel([9,2,3,4,5,6])
ax.set_ylabel("Month")
x=["acc","acc","acc","acc","acc","acc"]
#plt.xticks(range(6),["acc","acc","acc","acc","acc","acc"])
ax.set_xticklabels(x,ha='center')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.show()
"""
x=loadcsv("Data/pca-50.csv",' ')
svm=x[::2]
rf=x[1::2]
GDBT=x[2::2]
print(average(svm))
print(rf)
print(GDBT)
"""