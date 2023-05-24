"""
author:Xian Tan
"""
from op import from_pickle,to_pickle
from gensim.models import Word2Vec,FastText
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

def main():
    f=open("./Data/pos.fasta",'r')
    f2 = open("./Data/neg.fasta", 'r')
    lis2=fasta2seq(f2)
    print(len(lis2))
    lis2=cutseq(lis2,128)
    lis2=seq2kmer(lis2,3)
    lis=fasta2seq(f)
    print(len(lis))
    lis=cutseq(lis,128)
    lis=seq2kmer(lis,3)
    lis+=lis2
    print(len(lis))
    lis2=[]
    model=Word2Vec(sentences=lis,vector_size=16,min_count=1)
    model2=FastText(sentences=lis,vector_size=16)
    lis3=[]
    lis4=[]
    for i in lis:
        #print(len(model.wv[i]))
        lis3.append(model.wv[i])
    to_pickle(lis3,"word2vec_16")
    lis3=[]
    for i in lis:
        lis4.append(model2.wv[i])
    to_pickle(lis4, "fasttext_16")
    #print(model.wv['att',"ttt"])
    #similar_words = model.wv.most_similar('att')
    #print(similar_words)
    #print(model2.wv["ttt"])

def fasta2seq(file):
    """
    将fasta文件转化为序列列表
    :param file: fasta文件
    :return: 序列列表
    """
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



def cutseq_center(lis,k):
    """
    将序列列表中的序列按照k的大小切割
    :param lis: 序列列表
    :param k: kmer大小
    :return: kmer列表
    """
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

def seq2kmer(lis,k):
    """
    将序列列表中的序列转化为kmer列表
    :param lis: 序列列表
    :param k: kmer大小
    """
    lis2=[]
    for i in lis:
        lis_temp=[]
        for j in range(len(i)-k+1):
            lis_temp.append(i[j:j+k])
        lis2.append(lis_temp)
    return lis2

def seq2kmer_bert(lis,k):
    lis2=[]
    for i in lis:
        str_temp=""
        for j in range(len(i)-k+1):
            str_temp=str_temp+str(i[j:j+k])+" "
        lis2.append(str_temp.rstrip(" "))
    return lis2
            
if __name__=="__main__":
    main()




