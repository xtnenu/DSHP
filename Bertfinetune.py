import torch
from transformers import AutoTokenizer, AutoModel
from pretreatment import fasta2seq,seq2kmer,seq2kmer_bert
from op import to_pickle
from torch.utils.data import Dataset,DataLoader
from utils import bert_cut
import numpy as np
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertModel
from DNAbert import Dataset2
from sklearn.model_selection import train_test_split
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)


class MyModel(torch.nn.Module):
    def __init__(self,number):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('zhihan1996/DNA_bert_'+str(number))
        self.fc = torch.nn.Linear(768, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, input_ids):
        output = self.bert(input_ids=input_ids)
        output = self.fc(output.pooler_output)
        output = self.activation(output)
        return output

def fine_tune(kmer):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name="zhihan1996/DNA_bert_"+str(kmer)
    tokenizer = AutoTokenizer.from_pretrained(name)

    model=MyModel(kmer)
    #print(model)
    f = open("./Data/pos.fasta", 'r')
    f2 = open("./Data/neg.fasta", 'r')
    lis2 = fasta2seq(f2)
    lis = fasta2seq(f)
    lis += lis2
    label= [0.0] * 9310 + [1.0] * 4471
    X_train, X_test, y_train, y_test = train_test_split(lis, label, test_size=0.1, random_state=1)
    with open('bert_split_result.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    X_train = bert_cut(X_train, 500)
    X_train = seq2kmer_bert(X_train, kmer)
    y_train_temp=[]
    for i in y_train:
        for j in range(4):
            y_train_temp.append(i)
    y_train=y_train_temp
    input_ids = [tokenizer.encode(seq, add_special_tokens=True) for seq in X_train]
    input_ids = torch.tensor(input_ids)
    y_train=torch.tensor(y_train)
    data=MyDataset(input_ids,y_train)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dataloader = DataLoader(data, batch_size=64, shuffle=False)
    model.to(device)
    for epoch in range(5):
        for batch in dataloader:
            input_ids, labels = batch
            input_ids=input_ids.to(device)
            labels=labels.unsqueeze(dim=1)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(input_ids)
            loss = loss_fn(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()
    modelname="Bert"+str(kmer)+".pth"
    torch.save(model, modelname)

def get_embeddings(kmer):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name="zhihan1996/DNA_bert_"+str(kmer)
    tokenizer = AutoTokenizer.from_pretrained(name)
    with open('bert_split_result.pkl', 'rb') as f:
        #pickle.dump((X_train, X_test, y_train, y_test), f)
        X_train, X_test, y_train, y_test=pickle.load(f)
    X_test = bert_cut(X_test, 500)
    X_test = seq2kmer_bert(X_test, kmer)
    y_train_temp=[]
    for i in y_train:
        for j in range(4):
            y_train_temp.append(i)
    y_train=y_train_temp
    input_ids = [tokenizer.encode(seq, add_special_tokens=True) for seq in X_train]
    input_ids = torch.tensor(input_ids)
    y_train=torch.tensor(y_train)
    data=MyDataset(input_ids,y_train)


#main(6)
#main(5)
#main(4)
#main(3)
