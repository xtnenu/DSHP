import torch
from transformers import AutoTokenizer, AutoModel
from pretreatment import fasta2seq,seq2kmer,cutseq,seq2kmer_bert
from op import to_pickle

def embed_dna_sequences(sequences):
    """
    Embed DNA sequences using the DNABERT model.

    Args:
    - sequences: A list of DNA sequences.

    Returns:
    - A numpy array of shape (len(sequences), 768) containing the embeddings of the input sequences.
    """
    # Load the DNABERT model.
    model_name = "zhihan1996/DNA_bert_6"
    #model_name ="dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Convert the input sequences to a PyTorch tensor.
    input_ids = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]
    input_ids = torch.tensor(input_ids)

    # Embed the sequences using the DNABERT model.
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    return embeddings
def main():
    f=open("./Data/pos.fasta",'r')
    f2 = open("./Data/neg.fasta", 'r')
    lis2=fasta2seq(f2)
    lis=fasta2seq(f)
    lis+=lis2
    lis=cutseq(lis,256)
    lis=seq2kmer_bert(lis,6)
    X=embed_dna_sequences(lis)
    to_pickle(X,"DNAbert_6.pkl")
if __name__=="__main__":
    main()