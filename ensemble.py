from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, concatenate

# 定义模型输入
input_1 = Input(shape=(100, 1), name='input_1')
input_2 = Input(shape=(200, 1), name='input_2')
input_3 = Input(shape=(300, 1), name='input_3')
input_4 = Input(shape=(400, 1), name='input_4')
input_5 = Input(shape=(500, 1), name='input_5')

# 第一个子结构：输入input_1，使用1个卷积层和1个LSTM层
conv_1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_1)
pool_1 = MaxPooling1D(pool_size=2)(conv_1)
lstm_1 = LSTM(64)(pool_1)

# 第二个子结构：输入input_2，使用2个卷积层和1个LSTM层
conv_2 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_2)
conv_2 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv_2)
pool_2 = MaxPooling1D(pool_size=2)(conv_2)
lstm_2 = LSTM(128)(pool_2)

# 第三个子结构：输入input_3，使用3个卷积层和1个LSTM层
conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu')(input_3)
conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv_3)
conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv_3)
pool_3 = MaxPooling1D(pool_size=2)(conv_3)
lstm_3 = LSTM(256)(pool_3)

# 第四个子结构：输入input_4，使用4个卷积层和1个LSTM层
conv_4 = Conv1D(filters=256, kernel_size=3, activation='relu')(input_4)
conv_4 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv_4)
conv_4 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv_4)
conv_4 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv_4)
pool_4 = MaxPooling1D(pool_size=2)(conv_4)
lstm_4 = LSTM(512)(pool_4)

# 第五个子结构：输入input_5，使用5个卷积层和1个LSTM层
conv_5 = Conv1D(filters=512, kernel_size=3, activation='relu')(input_5)
conv_5 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv_5)
conv_5 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv_5)
conv_5 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv_5)
conv_5 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv_5)
pool_5 = MaxPooling1D(pool_size=2)(conv_5)
lstm_5 = LSTM(1024)(pool_5)

# 拼接五个子结构的输出
concat = concatenate([lstm_1, lstm_2, lstm_3, lstm_4, lstm_5])

# 添加Dropout层
drop = Dropout(0.5)(concat)

# 添加全连接层
dense = Dense(units=256, activation='relu')(drop)

# 最后的输出层，为二分类问题，使用sigmoid激活函数
output = Dense(units=1, activation='sigmoid')(dense)

# 构建模型
model = Model(inputs=[input_1, input_2, input_3, input_4, input_5], outputs=output)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])