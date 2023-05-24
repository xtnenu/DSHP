from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def pca(X,n_component=50):
    """
    PCA降维需要满足以下几个要求：

    1. 数据需要以特征为列、样本为行的形式存储，且数据应该是线性可分的。

    2. 数据要有比较高的相关性，即特征之间相关性较强，否则PCA处理后的结果可能不如原始数据。

    3. 数据需要进行标准化处理，即每个特征的均值为0，方差为1，以避免某些特征的重要性过高。

    4. 计算协方差矩阵时需要考虑样本量勉强大于特征数量。如果特征数量远大于样本数量，在计算协方差矩阵时可能会产生数值不稳定的问题。

    5. 选择保留的主成分数时需要根据可解释方差的贡献率选择，一般保留累计贡献率达到95%以上的主成分。
    """
    model = PCA(n_component)
    X_pca = model.fit_transform(X)
    return X_pca

def ICA(X,n_components=50):
    """
    1. 独立性假设：独立成分分析的前提假设是，混合信号是由若干个相互独立的信号线性组合得到的。在现实中，这个假设并不总是成立，因此需要对数据进行合理的预处理，以尽量满足独立性假设。

    2. 数据数量：独立成分分析通常需要足够数量的数据，以便准确地估计混合矩阵和独立分量。一般来说，需要的数据量与信号的维度和复杂度有关。

    3. 数据源数量：独立成分分析的原理是基于混合矩阵的非奇异性，要求混合矩阵必须是可逆的，也就是说，需要混合矩阵中的列数等于信号源的数量。如果信号源数量未知，可以使用一些估算方法进行求解。

    4. 独立分量数量的确定：独立成分分析需要确定所提取的独立分量的数量。如果确定数量过多或过少，可能会导致提取的结果不理想。因此，需要在合理范围内确定独立分量数量的上限和下限。

    """
    ica = FastICA(n_components=n_components, random_state=0)
    X_ica = ica.fit_transform(X)
    return X_ica

def LDA(X,y,n_components=50):
    """
    
    其使用条件如下：

    1. 数据集具有较好的线性可分性，即不同类别的数据在特征空间中可以被一个超平面分开。

    2. 数据集中的样本数量不小于特征数量。

    3. 数据集的协方差矩阵是满秩的，即每个特征都具有独立的信息。

    4. 数据集的类别之间具有相同的协方差矩阵，即不同类别的数据在特征空间中具有相同的分布模式。
 
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    return X_lda