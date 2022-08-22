"""
author:Xian Tan
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow import compat
from tensorflow import double
import random
from Pyfeat_v2 import loadcsv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping


def auroc(y_true, y_pred):
    return compat.v1.py_func(roc_auc_score, (y_true, y_pred), double)

def aps(y_true, y_pred):
    return compat.v1.py_func(average_precision_score, (y_true, y_pred), double)


def Conv1d(x,y,dim=0,epoch=50):
    layerlist=[]
    n=len(x[0])
    x=x.reshape(-1,n,1)
    layerlist.append(layers.Conv1D(32,64,input_shape=(n,1),activation='relu'))
    layerlist.append(layers.AveragePooling1D(8))
    layerlist.append(layers.Flatten())
    layerlist.append(layers.Dense(8,activation='relu'))
    layerlist.append(layers.Dense(1,activation='sigmoid'))
    model = keras.Sequential(layerlist)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x, y, epochs=epoch, batch_size=64,validation_split=0.1)
    model.save("Conv1d3" + '.h5')
    return model

def Loaddlmodel(x,y,path):
    n = len(x[0])
    x = x.reshape(-1, n, 1)
    model=load_model(path,custom_objects={'auroc': auroc})
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',auroc,aps])
    model.summary()
    result=model.evaluate(x,y,batch_size=128)
    return result

def Conv1d_cross(x,y,t_x,t_y,score,epoch=500):
    layerlist = []
    n = len(x[0])
    x = x.reshape(-1, n, 1)
    t_x=t_x.reshape(-1, n, 1)
    layerlist.append(layers.Conv1D(32, 64, input_shape=(n, 1), kernel_initializer="orthogonal",activation='relu'))
    layerlist.append(layers.AveragePooling1D(8))
    layerlist.append(layers.Flatten())
    layerlist.append(layers.Dropout(0.25))
    layerlist.append(layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layerlist)
    SGDN=SGD(learning_rate=0.0005)
    es = EarlyStopping(monitor='val_loss', patience=200, verbose=2)
    model.compile(loss='binary_crossentropy', optimizer=SGDN,metrics=['accuracy'])
    model.summary()
    model.fit(x, y, epochs=epoch, batch_size=32,validation_split=0.1)



    model.compile(loss='binary_crossentropy', optimizer=SGDN,metrics=['accuracy', auroc])

    result = model.evaluate(t_x, t_y,batch_size=128)
    score.append(result)
    return model

def Conv1d_cross_2(x,y,t_x,t_y,score,epoch=500):
    layerlist = []
    n = len(x[0])
    x = x.reshape(-1, n, 1)
    t_x=t_x.reshape(-1, n, 1)
    layerlist.append(layers.Conv1D(32, 64, input_shape=(n, 1), kernel_initializer="orthogonal",activation='relu'))
    layerlist.append(layers.AveragePooling1D(8))
    layerlist.append(layers.Flatten())
    layerlist.append(layers.Dropout(0.25))
    #layerlist.append(layers.Dense(8, activation='relu'))
    layerlist.append(layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layerlist)
    #AM=Adam(lr=0.0005)
    SGDN=SGD(learning_rate=0.0005)
    #RS=RMSprop(learning_rate=0.0005)
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
    model.compile(loss='binary_crossentropy', optimizer=SGDN,metrics=['accuracy',auroc,aps])
    model.summary()
    model.fit(x, y, epochs=epoch, batch_size=32,callbacks=[es],validation_split=0.1)

    model.compile(loss='binary_crossentropy', optimizer=SGDN,metrics=['accuracy', auroc,aps])
    #model.save("Conv1d3" + '.h5')

    result = model.evaluate(t_x, t_y,batch_size=128)
    score.append(result)
    return model

def Conv1d_cross_show(x,y,t_x,t_y,score,epoch=500):
    layerlist = []
    n = len(x[0])
    x = x.reshape(-1, n, 1)
    t_x=t_x.reshape(-1, n, 1)
    layerlist.append(layers.Conv1D(32, 64, input_shape=(n, 1), kernel_initializer="orthogonal",activation='relu'))
    layerlist.append(layers.AveragePooling1D(8))
    layerlist.append(layers.Flatten())
    layerlist.append(layers.Dropout(0.25))
    #layerlist.append(layers.Dense(8, activation='relu'))
    layerlist.append(layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layerlist)
    #AM=Adam(lr=0.0005)
    SGDN=SGD(learning_rate=0.0005)
    #RS=RMSprop(learning_rate=0.0005)
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
    model.compile(loss='binary_crossentropy', optimizer=SGDN,metrics=['accuracy'])
    model.summary()

    return model


def Crossvilidation(path,n=10,title="model"):
    b = loadcsv(path, ",")
    b = np.array(b)
    b = b.astype("float32")

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)

    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        model = Conv1d_cross_2(X[train], Y[train], X[test], Y[test], cvscores)
        model.save(title + str(i) + ".h5")
        i += 1
    print(cvscores)
    a = 0
    b = 0
    for i in cvscores:
        a = a + i[1]
        b = b + i[2]
    print(a / n, b / n)


def abl_nonzc(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][3:]
    return matrix
def abl_nongc(matrix):
    for j in range(len(matrix)):
        matrix[j]=matrix[j][0:3]+matrix[j][4:]
    return matrix
def abl_noncm(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][0:4] + matrix[j][6:]
    return matrix
def abl_non1(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][0:6] + matrix[j][7:]
    return matrix
def abl_non2(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][0:7] + matrix[j][91:]
    return matrix
def abl_nonoth(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][91:]
    return matrix
def abl_nonMono(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][0:91] + matrix[j][2091:8491]+matrix[j][9771:14892]
    return matrix
def abl_nonD(matrix):
    for j in range(len(matrix)):
        matrix[j]=matrix[j][0:171]+ matrix[j][491:1771]+matrix[j][8491:9771]+[matrix[j][-1]]
    return matrix

def abl_nontri(matrix):
    for j in range(len(matrix)):
        matrix[j] = matrix[j][0:491] + matrix[j][1771:3771]+[matrix[j][-1]]
    return matrix

def Cv_abl_show(path,n=10,title="model"):
    b = loadcsv(path, ",")
    if title=="nonzc":
        b = abl_nonzc(b)
    elif title=="nongc":
        b = abl_nongc(b)
    elif title=="noncm":
        b =abl_noncm(b)
    elif title=="non1":
        b=abl_non1(b)
    elif title=="nonkmer":
        b=abl_non2(b)
    elif title=="nonoth":
        b=abl_nonoth(b)
    elif title=="nonmo":
        b=abl_nonMono(b)
    elif title=="nond":
        b=abl_nonD(b)
    elif title=="nontri":
        b=abl_nontri(b)
    b = np.array(b)
    b = b.astype("float32")
    print(len(b[0])-1)

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)

    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        model = Conv1d_cross_show(X[train], Y[train], X[test], Y[test], cvscores)
    return model

def Cv_abl(path,n=10,title="model"):
    b = loadcsv(path, ",")
    if title=="nonzc":
        b = abl_nonzc(b)
    elif title=="nongc":
        b = abl_nongc(b)
    elif title=="noncm":
        b =abl_noncm(b)
    elif title=="non1":
        b=abl_non1(b)
    elif title=="nonkmer":
        b=abl_non2(b)
    elif title=="nonoth":
        b=abl_nonoth(b)
    elif title=="nonmo":
        b=abl_nonMono(b)
    elif title=="nond":
        b=abl_nonD(b)
    elif title=="nontri":
        b=abl_nontri(b)
    b = np.array(b)
    b = b.astype("float32")

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)

    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        model = Conv1d_cross_2(X[train], Y[train], X[test], Y[test], cvscores)
        model.save(title + str(i) + ".h5")
        i += 1
    print(cvscores)
    a = 0
    b = 0

    for i in cvscores:
        a = a + i[1]
        b = b + i[2]
    print(a / n, b / n)


def testtemp(path,n=10):
    b = loadcsv(path, ",")
    b = np.array(b)
    b = b.astype("float32")

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)

    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        x=Loaddlmodel(X[test], Y[test],"model" + str(i) + ".h5")
        cvscores.append(x)
        i += 1
    a=0
    b=0
    for i in cvscores:
        a=a+i[1]
        b=b+i[2]
    print(a/n,b/n)

def testtemp2(path,title,n=10):
    b = loadcsv(path, ",")
    if title=="nonzc":
        b = abl_nonzc(b)
    elif title=="nongc":
        b = abl_nongc(b)
    elif title=="noncm":
        b =abl_noncm(b)
    elif title=="non1":
        b=abl_non1(b)
    elif title=="nonkmer":
        b=abl_non2(b)
    elif title=="nonoth":
        b=abl_nonoth(b)
    elif title=="nonmo":
        b=abl_nonMono(b)
    elif title=="nond":
        b=abl_nonD(b)
    elif title=="nontri":
        b=abl_nontri(b)
    b = np.array(b)
    b = b.astype("float32")

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)

    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        x=Loaddlmodel(X[test], Y[test],title + str(i) + ".h5")
        cvscores.append(x)
        i += 1
    a=0
    b=0
    c=0
    for i in cvscores:
        a=a+i[1]
        b=b+i[2]
        c=c+i[3]
    print(a/n,b/n,c/n)



if __name__=="__main__":

    Crossvilidation(["Data/Data.csv"],n=10,title="model")
