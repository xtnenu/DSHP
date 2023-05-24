import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# 定义3-mer DNA Transformer模型
"""
def create_transformer_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs
    # 嵌入层，将每个字符嵌入到512维的向量空间
    x = layers.Embedding(input_dim=64, output_dim=512)(x)
    # 位置编码器，用于将序列中每个位置的嵌入向量进行编码
    position_embedding = layers.Embedding(input_dim=10000, output_dim=512)(tf.range(start=0, limit=input_shape[0], delta=1))
    x = layers.Add()([x, position_embedding])
    # Transformer编码器
    for i in range(6):
        x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = x
        x = layers.Conv1D(filters=2048, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Conv1D(filters=512, kernel_size=1, activation=None)(x)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = x
        x = layers.Conv1D(filters=2048, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Conv1D(filters=512, kernel_size=1, activation=None)(x)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x1 = x
        x = layers.Conv1D(filters=2048, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Conv1D(filters=512, kernel_size=1, activation=None)(x)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    # 全局平均池化层，用于将每个位置的向量进行平均池化
    x = layers.GlobalAveragePooling1D()(x)
    # 全连接层
    x = layers.Dense(units=2048, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    # 构建模型
    model = keras.Model(inputs=inputs, outputs=outputs, name="DNATransformer")
    return model
import numpy as np
# 生成随机的3-mer DNA序列数据
input_data = np.random.randint(0, 64, size=(1000, 100))
# 生成随机的标签数据
label_data = np.random.randint(0, 2, size=(1000,))
# 创建模型
model = create_transformer_model(input_shape=(100,), num_classes=2)
# 编译模型
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 训练模型
model.fit(input_data, label_data, batch_size=32, epochs=10, validation_split=0.2)
"""
from gensim.models import Word2Vec
import numpy as np
import itertools
# 生成所有可能的3-mer序列
def dna_to_3mer(dna_sequence):
    # 将 DNA 序列按照每三个碱基一个为一组进行切片，得到所有 3-mer
    return [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]

def dna_to_3mer(dna_sequence):
    # 将 DNA 序列按照每三个碱基一个为一组进行切片，得到所有 3-mer
    list=[]
    for j in dna_sequence:
        k = 3
        list.append([j[i:i + k] for i in range(len(j) - k + 1)])
    return list

nucleotides = ['A', 'C', 'G', 'T']
kmer_list = [''.join(p) for p in itertools.product(nucleotides, repeat=3)]
print(kmer_list)
# 构建3-mer序列列表
sequences = ['AAAAGGGGTTTTCCCC', 'GGGTTTAAACCCGGGTTTAAACCC',"AATTACGACACTAGA"]
sequences = dna_to_3mer(sequences)
print(sequences)
# 构建word2vec模型
embedding_size = 32
window_size = 5
model = Word2Vec(sentences=[list(seq) for seq in sequences], vector_size=embedding_size, window=window_size, min_count=1, workers=4)
# 获取每个3-mer序列的嵌入向量
embedding_matrix = np.zeros((len(kmer_list), embedding_size))
print(model.wv.index_to_key)
for i, kmer in enumerate(kmer_list):
    embedding_matrix[i] = model.wv[kmer]
# 打印嵌入矩阵的形状
print(embedding_matrix.shape)