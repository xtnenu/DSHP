from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,confusion_matrix,accuracy_score,recall_score,matthews_corrcoef


def binary(list):
    list2=[]
    for i in list:
        if i>=0.5:
            list2.append(1)
        else:
            list2.append(0)
    return list2

def auroc(y_true, y_pred):
    return compat.v1.py_func(roc_auc_score, (y_true, y_pred), double)

def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
def aps(y_true, y_pred):
    return compat.v1.py_func(average_precision_score, (y_true, y_pred), double)

def aupr(y_true, y_pred):
    return average_precision_score(y_true, y_pred)
def acc(y_true, y_pred):
    y_pred = binary(y_pred)
    acc = accuracy_score(y_true, y_pred)
    return acc

def pre(y_true, y_pred):
    y_pred=binary(y_pred)
    pre = precision_score(y_true, y_pred)
    return pre
def sen(y_true, y_pred):
    y_pred=binary(y_pred)
    sen = recall_score(y_true, y_pred)
    return sen
def mcc(y_true, y_pred):
    y_pred = binary(y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc
def spe(y_true, y_pred):
    y_pred = binary(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp)
    return spe

#y=[1,1,1,1,1,0]
#pred=[0,0,0,1,1,0]
#print(confusion_matrix(y,pred).ravel())#TN,FP,TP,FN