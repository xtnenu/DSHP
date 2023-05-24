"""
author:Xian Tan

"""


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score
from ilearn_f import Descriptor
#import keras
import re

class Framework:
    def __init__(self,path,length,al='rf',ilkw=None):
        self.path=path
        self.length=length
        self.al=al
        self.ilkw={
            'sliding_window': 5,
            'kspace': 3,
            'props': ['CIDH920105', 'BHAR880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 'BIGC670101', 'CHAM810101',
                      'DAYM780201'],
            'nlag': 3,
            'weight': 0.05,
            'lambdaValue': 3,
            'PseKRAAC_model': 'g-gap',
            'g-gap': 2,
            'k-tuple': 2,
            'RAAC_clust': 1,
            'aaindex': 'ANDN920101;ARGP820101;ARGP820102;ARGP820103;BEGF750101;BEGF750102;BEGF750103;BHAR880101',
            'kmer': 3,
            'mismatch': 1,
            'delta': 0,
            'Di-DNA-Phychem': 'Twist;Tilt;Roll;Shift;Slide;Rise',
            'Tri-DNA-Phychem': 'Dnase I;Bendability (DNAse)',
            'Di-RNA-Phychem': 'Rise (RNA);Roll (RNA);Shift (RNA);Slide (RNA);Tilt (RNA);Twist (RNA)',
            'distance': 0,
            'cp': 'cp(20)',}
        if ilkw!=None:
            for i in ilkw:
                self.ilkw[i]=ilkw[i]

    def loadtxt(self):
        pass

    def loadfasta(self):
        fasta_sequences = []
        labels=[]
        for i in self.path:
            with open(i) as f:
                records = f.read()
            records = records.split('>')[1:]

            for fasta in records:
                array = fasta.split('\n')
                header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWY-]', '-', ''.join(array[1:]).upper())
                header_array = header.split('|')
                label = header_array[1]
                labels.append(label)
                fasta_sequences.append(sequence)
        return fasta_sequences, labels
        

    def loaddata(self,al=['Kmer']):#.npy/fasta 2 csv
        if ".npy" in self.path[0].lower():
            train_data=np.load(self.path[0])
            train_label=np.load(self.path[1])
            test_data=np.load(self.path[2])
            test_label=np.load(self.path[3])
            return train_data,train_label,test_data,test_label
        elif ".fasta" in self.path[0].lower():
            result=[]
            data_dic={}
            for i in self.path:
                Des=Descriptor(i,self.ilkw)
                for j in al:
                    print(j)
                    getattr(Des,j)()
                    if j not in data_dic:
                        data_dic[j]=Des.get_data()
                    else:
                        data_dic[j]=np.append(data_dic[j],Des.get_data(),axis=0)
            
            for j in data_dic:
                np.random.seed(0)
                np.random.shuffle(data_dic[j])
                data=data_dic[j][:,2:].astype(float)
                lable=data_dic[j][:,1].astype(int)
                result.append((data,lable))
            return result
                    
        

    def split(self,data,pos=0,sp_r=0.1):
        le=len(data)
        if pos+int((1-sp_r)*le)<le:
            b=data[pos:pos+int((1-sp_r)*le)]
            if pos!=0:
                a=np.append(data[0:pos],data[pos+int((1-sp_r)*le):],axis=0)
            else:
                a=data[pos+int((1-sp_r)*le):]
        return (a,b)

    def clsf_rf(**kw):
        model=RandomForestClassifier(**kw)
        return model

    def clsf_svm(**kw):
        model=svm.SVC(**kw)
        return model

    def clsf_xgb(**kw):
        pass
        

    def algorithm(self,clf,**kw):#Match selected classifier
        func_name="clsf_"+clf
        func=getattr(Framework,func_name)
        model=func(**kw)

        return model

    def voting(self,lis):
        aim=0
        for i in lis:
            if i>0.5:
                aim+=1
        if aim>len(lis)/2:
            return 1
        else:
            return 0
    
    def vote(self,lis):
        index=0
        result=[]
        for i in range(len(lis[0])):
            temp=0
            for j in lis:
                temp+=j[index]
            if temp>=int(len(lis)/2+0.5):
                result.append(1)
            else:
                result.append(0)
            index+=1
        return result

    def vote_prob(self,lis):
        index=0
        result=[]
        for i in range(len(lis[0])):
            temp=0.0
            for j in lis:
                temp+=j[index][1]
            if temp>=len(lis)/2:
                result.append(1)
            else:
                result.append(0)
            index+=1
        return result

    def knn(self,lis):
        pass
                
    
    def ensemble(self,al=['Kmer'],model_clf='rf',mid_put=0,**kw):
        models=[]
        vote=[]
        if isinstance(model_clf,str)==1:
            data=self.loaddata(al=al)
            for i in data:
                x_train,x_test=self.split(i[0])
                y_train,y_test=self.split(i[1])
                model=self.algorithm(clf=model_clf,**kw)
                model.fit(x_train,y_train)
                if mid_put==0:
                    result=model.predict(x_test)
                else:
                    result=model.predict_proba(x_test)
                print(result)
                vote.append(result)
                print(y_test)
                if mid_put==0:
                    print(accuracy_score(y_test,result))
                
                models.append(model)
            print("----------------------------")
            if mid_put==0:
                ensemble_result=self.vote(vote)
            else:
                ensemble_result=self.vote_prob(vote)
            print(accuracy_score(y_test,ensemble_result))
            

        return models

    def train_single(self,al=['Kmer'],model_clf='rf',**kw):
        data_con=[]
        data=self.loaddata(al=al)
        for i in data:
            if data_con==[]:
                data_con=i[0]
            else:
                data_con=np.append(data_con,i[0],axis=1)
        label=data[0][1]
        data=0
        x_train,x_test=self.split(data_con)
        y_train,y_test=self.split(label)
        model=self.algorithm(clf=model_clf,**kw)
        model.fit(x_train,y_train)
        result=model.predict(x_test)
        print(result)
        print(accuracy_score(y_test,result))
        """
        print(label)
        print(data_con)
        """
    """
    def feature(self,train_data,train_label,test_data,test_label,style='normal'):
        if style=='normal':
            return train_data,train_label,test_data,test_label
        else:
            pass
    """
    def train(self,algorithm=0,savepath=None,feature_style=0):
        train_data,train_label,test_data,test_label=self.loaddata()
        train_data,train_label,test_data,test_label=self.feature(train_data,train_label,test_data,test_label,feature_style)
        train_data=train_data.reshape(len(train_data),8000)
        
        train_label=train_label.reshape(len(train_data),1)
        model=self.algorithm()
        model.fit(train_data,train_label)
        if savepath!=None:
            pass
        return model
        

    def test(self,model,test_data,test_label):
        pred=model.predict(test_data)
        
        



if __name__=="__main__":
    """
    a=Framework(["data//VISDB_Test_Data.npy",
                 "data//VISDB_Test_Label.npy",
                 "data//dsVIS_Test_Data.npy",
                 "data//dsVIS_Test_Label.npy"],
                 0)
    a.train()
    """
    #'NAC','EIIP'
    al=['Kmer','RCKmer','Z_curve_144bit','Z_curve_12bit','Z_curve_36bit','Z_curve_48bit']
    a=Framework(['data//neg.fasta','data//test.fasta'],2000)
    
    a.ensemble(al=al,mid_put=0,n_estimators=200)
    #a.train_single(al=['Kmer','RCKmer','Z_curve_9bit','NAC','EIIP',],model_clf='rf',n_estimators=200)

