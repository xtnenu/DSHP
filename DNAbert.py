import torch
from transformers import AutoTokenizer, AutoModel
from pretreatment import fasta2seq,seq2kmer,seq2kmer_bert
from op import to_pickle
from torch.utils.data import Dataset,DataLoader
from utils import bert_cut
import numpy as np
class MyDataset(Dataset):
    """
    自定义数据集类
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

class Dataset2(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

def Finetune(data):

    # 加载模型和预训练分词模型
    tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
    model = AutoModel.from_pretrained('zhihan1996/DNA_bert_6')

    #  加载自己的数据集，准备输入模型进行微调
    dataset = data# 自己的数据集，需要进行处理以符合模型的输入格式

    #  对数据集进行分批并输入模型进行微调
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs, labels=labels)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained('path/to/your/fine-tuned/dna-bert/model')
    tokenizer.save_pretrained('path/to/your/fine-tuned/dna-bert/tokenizer')

def embed_dna_sequences(sequences,model_name="zhihan1996/DNA_bert_6"):
    """
    Embed DNA sequences using the DNABERT model.

    Args:
    - sequences: A list of DNA sequences.

    Returns:
    - A numpy array of shape (len(sequences), 768) containing the embeddings of the input sequences.
    """
    # Load the DNABERT model.
    #model_name = "zhihan1996/DNA_bert_6"
    #model_name ="dmis-lab/biobert-v1.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Convert the input sequences to a PyTorch tensor.
    input_ids = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]
    input_ids = torch.tensor(input_ids)
    dataset = Dataset2(input_ids)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # Embed the sequences using the DNABERT model.
    n=np.array([])
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            if n.size==0:
                n=embeddings
            else:
                n=np.concatenate([n,embeddings],axis=0)

    return embeddings
def main(n=6):
    f=open("./Data/pos.fasta",'r')
    f2 = open("./Data/neg.fasta", 'r')
    lis2=fasta2seq(f2)
    print(len(lis2))
    lis=fasta2seq(f)
    print(len(lis))
    lis+=lis2
    lis=bert_cut(lis,500)
    model_name="zhihan1996/DNA_bert_"+str(n)
    #lis=cutseq(lis,128)
    #lis=[[seq[0:6]+" "+seq[1:7]+" "+seq[2:8]] for seq in lis]
    #lis=[seq[0:6]" "+seq[1:7]+" "+seq[2:8] for seq in lis]
    lis=seq2kmer_bert(lis,n)
    X=embed_dna_sequences(lis,model_name=model_name)
    #to_pickle(X,"Biobert.pkl")
    name="DNAbert"+str(n)+"new"
    to_pickle(X,name)

if __name__=="__main__":
    main(3)
    main(4)
    main(5)
    main(6)