
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
#from sklearn.ensemble import
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error, r2_score
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow import compat
from tensorflow import double
import random
from Pyfeat_v2 import loadcsv,loadcsv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import visualize as vis
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from op import from_pickle
from Metrics import acc,mcc,pre,auc,sen,spe,aupr
from Dimensionality_Reduction import pca

def SVM(X,y):#构建第一个函数
    model = svm.SVC()
    model.fit(X, y)
    return model


def SVM2():
    X = [[0, 0, 0, 0], [1, 1, 0, 0],[2, 1, 0, 0],[0, 1, 0, 0]]
    y = [0, 1, 1, 0]
    model = svm.SVC()
    model.fit(X, y)
    return model

def RF(X_train, y_train):
    rf_model=RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

def GBDT(X=[[0, 0], [1, 1], [2, 2], [3, 3]],y=[0, 0, 1, 1]):
    """
    `n_estimators`: 迭代次数，默认为100
    `learning_rate`: 学习率，每个决策树的贡献缩小的速度，默认为1.0
    `max_depth`: 每个决策树的最大深度，默认为3
    `random_state`: 随机数种子，设置相同的种子可以保证每次运行训练结果一致，默认为None。
    """
    # 创建分类器
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
    # 训练模型
    clf.fit(X, y)
    # 预测
    #clf.predict([[2.5, 2.5], [0.5, 0.5]])
    return clf

def Xgboost(X,y):
    # 加载数据
    #data, label = load_breast_cancer(return_X_y=True)

    # 划分训练集和测试集
    #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

    # 构造xgboost数据矩阵
    #dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    #dtest = xgb.DMatrix(Xtest, label=ytest)

    # 设置参数
    param = {
        'max_depth': 3,  # 树的最大深度
        'eta': 0.1,  # 学习率
        'objective': 'binary:logistic',  # 损失函数
        'eval_metric': 'auc',  # 评价指标
        #'silent': 1  # 是否输出中间过程
    }
    dtrain=xgb.DMatrix(X,label=y)
    # 训练模型
    num_round = 50  # 迭代次数
    bst = xgb.train(param, dtrain, num_round)
    return bst
    # 预测测试集样本
    #ypred = bst.predict(dtest)

    # 计算AUC
    #print('AUC: {:.2f}'.format(roc_auc_score(ytest, ypred)))


def main(path,n=10,Pca=0):
    if ".csv" in path[0]:
        X = loadcsv(path, ",")
        X = np.array(X)
        X = X.astype("float32")
        np.random.seed(0)
        np.random.shuffle(X)
        Y = X[:, -1]
        X = X[:, :-1]
        if Pca!=0:
            X = pca(X,Pca)

    elif ".pkl" in path:
        X = from_pickle(path)
        print(len(X))
        lst = [0] * 9310 + [1] * 4471
        X=np.array(X)
        X=X.reshape(len(X),-1)
        X=pca(X)
        Y = np.array(lst)
        Y=Y.reshape(len(Y),1)
        #X=X.reshape(len(Y),4032)
        print(X.shape,Y.shape)
        X = np.hstack((X, Y))
        np.random.shuffle(X)
        Y = X[:,-1]
        X = X[:, :-1]
        if Pca!=0:
            X=pca(X,Pca)

    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        #model = Conv1d_cross_3(X[train], Y[train], X[test], Y[test], cvscores)
        model_svm=SVM(X[train], Y[train])
        model_rf=RF(X[train], Y[train])
        model_gbdt=GBDT(X[train], Y[train])
        #model_xgboost=
        clf_list=[model_svm,model_rf,model_gbdt]
        for j in clf_list:
            pred=j.predict(X[test])
            clf_acc=acc(Y[test],pred)
            clf_aupr=aupr(Y[test],pred)
            clf_auc=auc(Y[test],pred)
            clf_pre=pre(Y[test],pred)
            clf_sen=sen(Y[test],pred)
            clf_spe=spe(Y[test],pred)
            clf_mcc=mcc(Y[test],pred)
            #print("acc:",clf_acc,"auc:",clf_auc,clf_pre,clf_sen,clf_spe,clf_mcc,clf_aupr)
            print("acc:", clf_acc, "auc:", clf_auc, "aupr:",clf_aupr)
        #vis.struc_vis(model,title+str(i)+".pdf")

        #model.save(title + str(i) + ".h5")
        #i += 1
    #print(cvscores)
    #a = 0
    #b = 0
    #for i in cvscores:
        #a = a + i[1]
        #b = b + i[2]
    #f = open("records.txt", 'w')
    #print(a / n, b / n, file=f)

def main_x(path,n=10,Pca=0):
    if ".csv" in path[0]:
        X = loadcsv(path, ",")
        X = np.array(X)
        X = X.astype("float32")
        np.random.seed(0)
        np.random.shuffle(X)
        Y = X[:, -1]
        X = X[:, :-1]
        if Pca!=0:
            X = pca(X,Pca)

    elif ".pkl" in path:
        X = from_pickle(path)
        print(len(X))
        lst = [0] * 9310 + [1] * 4471
        X=np.array(X)
        X=X.reshape(len(X),-1)
        if Pca!=0:
            X=pca(X,n_component=Pca)
        Y = np.array(lst)
        Y=Y.reshape(len(Y),1)
        #X=X.reshape(len(Y),4032)
        print(X.shape,Y.shape)
        X = np.hstack((X, Y))
        np.random.shuffle(X)
        Y = X[:,-1]
        X = X[:, :-1]

    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    cvscores = []
    i = 0
    for train, test in kfold.split(X, Y):
        #model = Conv1d_cross_3(X[train], Y[train], X[test], Y[test], cvscores)
        #model_svm=SVM(X[train], Y[train])
        #model_rf=RF(X[train], Y[train])
        #model_gbdt=GBDT(X[train], Y[train])
        model_xgboost=Xgboost(X[train], Y[train])
        #clf_list=[model_svm,model_rf,model_gbdt]
        clf_list=[model_xgboost]
        for j in clf_list:
            pred=j.predict(xgb.DMatrix(X[test],label=Y[test]))
            clf_acc=acc(Y[test],pred)
            clf_aupr=aupr(Y[test],pred)
            clf_auc=auc(Y[test],pred)
            clf_pre=pre(Y[test],pred)
            clf_sen=sen(Y[test],pred)
            clf_spe=spe(Y[test],pred)
            clf_mcc=mcc(Y[test],pred)
            print(clf_acc,clf_auc,clf_pre,clf_sen,clf_spe,clf_mcc,clf_aupr)


if __name__=="__main__":#这里是python的一个独有机制，就是将本文件作为主文件运行时可以触发下面的代码，如果觉得理解起来有点困难可以暂时作为“固定搭配”记一下，用多了就理解了
    #a=SVM()#这里的a就是由第一个函数训练好的模型
    #result=a.predict([[1,0],[-1,0]])#模型拥有一些写好的方法，这里的predict就是根据已有的模型去预测了两条未知的数据，注意未知数据与训练数据的 向量维度 要统一。
    #print(result)

    #b=SVM2()
    #result=b.predict([[1, 0, 0, 0],[-1, 0, 0, 0]])
    #trueLabel=[0,0]#这里不同于上一部分，我给出了真实标签。和上一行的两条数据一起构成了一个样本量为2的测试集，上一行是测试集数据、这一行是测试集标签。
    #print(result)
    #print(metrics.accuracy_score(trueLabel,result))#因为有了完整的测试集，这里调用了一个评价模型的函数,算accuracy值的
    #main(["Data.csv"],10)
    #main(["Data.csv"],10)
    """
    main(["Data.csv"],10,Pca=50)
    print("50finish")
    main(["Data.csv"],10,Pca=200)
    print("200finish")
    main(["Data.csv"], 10, Pca=2000)
    """
    #main("DNAbert_6_256.pkl.pkl",10)
    #main("Biobert.pkl",10)
    #main("fasttext_16.pkl",10)
    #main("word2vec_16.pkl",10)
    #main("DNA3.pkl",10)
    #main("DNA4.pkl", 10)
    #main("DNA5.pkl", 10)
    #main("DNA6.pkl", 10)

    main_x(["Data.csv"], 10)
    main_x(["Data.csv"], 10, Pca=200)
    main_x(["Data.csv"], 10, Pca=2000)
    main_x(["Data.csv"], 10, Pca=5000)
    #main_x("DNA3.pkl",10)
    #main_x("DNA4.pkl",10)
    #main_x("DNA5.pkl",10)
    #main_x("DNA6.pkl",10)

