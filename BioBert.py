import numpy as np
from biobert_embedding.embedding import BiobertEmbedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pretreatment
# 加载DNA序列数据
f=open("./Data/pos.fasta",'r')
data = pretreatment.fasta2seq(f)

#labels = np.load('dna_labels.npy')

# 划分训练集和测试集
#x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 使用BioBERT提取DNA序列的词向量
biobert = BiobertEmbedding()
x = biobert.embed_sentences(data)
for i in x:
    print(i)

# 构建分类器
#input_shape = (x_train.shape[1],)
#model_input = Input(shape=input_shape)
#x = Dense(64, activation='relu')(model_input)
#x = Dense(32, activation='relu')(x)
#model_output = Dense(1, activation='sigmoid')(x)
#model = Model(inputs=model_input, outputs=model_output)

# 编译和训练分类器
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))