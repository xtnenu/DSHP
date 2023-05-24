from tensorflow.keras.models import Model,load_model
#import umap.umap_ as umap
import umap
import numpy as np
from utils import loadcsv,from_pickle
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
def Crossvilidation(path,n=10):
    if ".csv" in path:
        X = loadcsv(path, ",")
        X = np.array(X)
        X = X.astype("float32")
        np.random.seed(0)
        np.random.shuffle(X)
        Y = X[:, -1]
        X = X[:, :-1]
    elif ".pkl" in path:
        X = from_pickle(path)
        print(len(X))
        lst = [0] * 9310 + [1] * 4471
        X=np.array(X)
        Y = np.array(lst)
        Y=Y.reshape(len(Y),1)
        #X=X.reshape(len(Y),4032)
        print(X.shape,Y.shape)
        X = np.hstack((X, Y))
        np.random.shuffle(X)
        Y = X[:,-1]
        X = X[:, :-1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    for train, test in kfold.split(X, Y):
        reducer=umap.UMAP(n_neighbors=2)
        embedding = reducer.fit_transform(X[test])
        plt.figure(figsize=(6, 6))
        # sns.scatterplot(data, hue=target_stack['test'], palette='Set1', sizes=2)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=Y[test], cmap='brg')
        plt.gca().set_aspect('equal', 'datalim')
        title = 'Merged embedding'

        plt.title(title)
        plt.xlabel('umap1')
        plt.ylabel('umap2')
        filename ="umap.svg"
        plt.savefig(filename)
        plt.show()
def Crossvilidation2(path,n=10):
    if ".csv" in path:
        X = loadcsv(path, ",")
        X = np.array(X)
        X = X.astype("float32")
        np.random.seed(0)
        np.random.shuffle(X)
        Y = X[:, -1]
        X = X[:, :-1]
    elif ".pkl" in path:
        X = from_pickle(path)
        print(len(X))
        lst = [0] * 9310 + [1] * 4471
        X=np.array(X)
        Y = np.array(lst)
        Y=Y.reshape(len(Y),1)
        #X=X.reshape(len(Y),4032)
        print(X.shape,Y.shape)
        X = np.hstack((X, Y))
        np.random.shuffle(X)
        Y = X[:,-1]
        X = X[:, :-1]
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    for train, test in kfold.split(X, Y):
        model=load_model("v2_new0.h5")
        model.summary()
        #model=Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        d=len(X[test][0])
        new=X[test].reshape(-1, 14891, 1)
        output = model.predict(new)
        output=output.reshape(len(output),-1)
        reducer=umap.UMAP(n_neighbors=2)
        #embedding = reducer.fit_transform(output)
        embedding = reducer.fit_transform(X[test])
        plt.figure(figsize=(6, 6))
        # sns.scatterplot(data, hue=target_stack['test'], palette='Set1', sizes=2)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=Y[test], cmap='brg')
        plt.gca().set_aspect('equal', 'datalim')
        title = ''
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18
        plt.tick_params(axis='both', labelsize=15)
        plt.title(title)
        plt.xlabel(' ')
        plt.ylabel(' ')
        filename ="umap2.svg"
        plt.savefig(filename)
        plt.show()
#model=load_model("v2_new0.h5")
#model.summary()
#modelx=Model(inputs=model.input, outputs=model.get_layer('conv1d').output)

#output = model.predict(input_data)

#reducer = umap.UMAP()
#embedding = reducer.fit_transform(output)
Crossvilidation2("Data.csv")