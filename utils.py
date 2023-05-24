import  pickle
import numpy as np
def fasta2seq(file):

    lis=[]
    seq=""
    for i in file:
        if ">" in i:
            if seq!="":
                #print(seq)
                lis.append(seq)
                seq=""
        else:
            i=i.rstrip("\n")
            seq+=i
    lis.append(seq)
    return lis

def to_pickle(lis,path):

    with open(path+".pkl", 'wb') as f:
        pickle.dump(lis, f)

def from_pickle(path):

    with open(path,'rb') as f:
        lis=pickle.load(f)
    return lis


def cutseq_center(lis,k):

    lis2=[]
    if len(lis[0])>k:
        for i in lis:
            if k%2==1:
                left=int(len(i)/2-0.5)-int((k-1)/2+0.5)-1
                right=int(len(i)/2-0.5)+int((k-1)/2+0.5)
                lis2.append(i[left:right])
            else:
                left=int(len(i)/2-0.5)-int((k-1)/2+0.5)
                right=int(len(i)/2-0.5)+int((k-1)/2+0.5)
                lis2.append(i[left:right])
    return lis2

def loadcsv_list(path,symbol):
    X=[]
    for i in path:
        file=open(i,"r")
        for j in file:
            X.append(j.rstrip("\n").split(symbol))
    return X

def loadcsv(path,symbol):
    X = []
    file = open(path, "r")
    for j in file:
        X.append(j.rstrip("\n").split(symbol))
    return X

def split_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def split_str_in_list(str_list, sub_len):

    sub_str_list = []
    for string in str_list:
        sub_str_list.append([string[i:i+sub_len] for i in range(0, len(string), sub_len)])
    return sub_str_list

def bert_cut(str_list, sub_len):
    lis=split_str_in_list(str_list, sub_len)
    lis=[j for i in lis for j in i]
    return lis

def bert_cont(path):
    lst=from_pickle(path)
    lst=np.array(lst)

    lst=lst.reshape(-1,3072)
    return lst
