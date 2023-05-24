import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
#from tensorflow.keras.applications import preprocess_input
from tensorflow.keras.utils import plot_model

#import netron
#from keras.models import load_model
#model = load_model('path/to/model.h5')
#netron.start(model)

# 获取模型中的卷积层
#plt.savefig('figure.eps', format='eps')
def visualize_features(model, x):
    # 获取模型中的卷积层
    conv_layer = model.get_layer('conv_layer')
    # 获取卷积层的权重
    weights = conv_layer.get_weights()[0]
    # 将权重可视化为图像
    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(8, 16))
    for i in range(32):
        axs[i//4, i%4].imshow(weights[:,:,0,i], cmap='gray')
        axs[i//4, i%4].axis('off')
    #plt.show()
    plt.savefig('figure.eps', format='eps')

    # 将卷积后的特征图可视化为图像
    # 获取模型中的卷积层
    conv_layer = model.get_layer('conv_layer')
    # 创建一个新的模型，只包含卷积层和输入层
    new_model = Model(inputs=model.inputs, outputs=conv_layer.output)
    # 对输入数据进行预处理
    x = preprocess_input(x)
    # 将输入数据输入到新模型中，得到卷积后的特征图
    features = new_model.predict(x)
    # 将特征图可视化为图像
    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(8, 16))
    for i in range(32):
        axs[i//4, i%4].imshow(features[0,:,:,i], cmap='gray')
        axs[i//4, i%4].axis('off')
    #plt.show()
    plt.savefig('figure.eps', format='eps')

from tensorflow.keras.models import load_model
#from vis.visualization import visualize_activation
#from vis.utils import utils
"""
def activation_vis():
    # 加载模型
    model = load_model('path/to/model.h5')
    # 获取模型中的卷积层
    layer_idx = utils.find_layer_idx(model, 'conv_layer')
    # 可视化激活
    img = visualize_activation(model, layer_idx, filter_indices=0)
    plt.imshow(img)
    plt.show()
"""

def plot_loss(history):
    # 获取训练集和验证集的损失值
    """
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    # 获取训练集和验证集的准确率
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # 获取训练轮数
    epochs = range(1, len(train_loss) + 1)

    # 绘制损失值曲线
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig('figure.eps', format='eps')

def struc_vis(model,name):
    plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradients(model, layer_name, input_image, class_index, colormap='viridis'):
    # 获取指定层的输出
    layer_output = model.get_layer(layer_name).output
    # 计算输出相对于输入的梯度
    grads = K.gradients(layer_output, input_image)[0]
    # 计算梯度的平均值
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # 定义一个函数，用于获取指定输入图像的梯度和输出特征图
    iterate = K.function([model.input], [pooled_grads, layer_output])
    # 获取梯度和输出特征图
    pooled_grads_value, layer_output_value = iterate([input_image])
    # 将梯度乘以输出特征图
    for i in range(layer_output_value.shape[-1]):
        layer_output_value[:, :, i] *= pooled_grads_value[i]
    # 计算输出特征图的通道均值
    heatmap = np.mean(layer_output_value, axis=-1)
    # 将通道均值归一化到0~1之间
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # 将热力图可视化为彩色图像
    plt.imshow(heatmap, cmap=colormap)
    plt.axis('off')
    plt.show()

    # 使用示例
    model = load_model('path/to/model.h5')
    input_image = load_image('path/to/image.jpg')
    visualize_gradients(model, 'conv_layer', input_image, 0)