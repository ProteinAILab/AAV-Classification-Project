import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import Callback
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from keras.initializers import RandomNormal, LecunNormal, Orthogonal

from keras.models import load_model
# from utils_f4fs import *
from model_inference_utils import limit_gpu_memory
from find_best_models import find_best_models2
from utils_f4f import CustomEarlyStopping, AA_hotencoding,  SaveModelCallback,SaveModelCallback_class
import shutil
limit_gpu_memory(gb_limits=[2, 2, 2], memory_limit_percent=None)

AA_col = 'AA'
output_folder = r'./fit4function_library_screens'
train = pd.read_csv(os.path.join(output_folder, 'train_data.csv'), usecols=['AA', 'is_viable'])
validate = pd.read_csv(os.path.join(output_folder, 'val_data.csv'), usecols=['AA', 'is_viable'])
test = pd.read_csv(os.path.join(output_folder, 'test_data.csv'), usecols=['AA', 'is_viable'])

train_x =  np.asarray([AA_hotencoding(variant) for variant in train[AA_col]])
validate_x = np.asarray([AA_hotencoding(variant) for variant in validate[AA_col]])
test_x = np.asarray([AA_hotencoding(variant) for variant in test[AA_col]])

# train_y = train.is_viable
# validate_y = validate.is_viable
# test_y = test.is_viable

train_y = train.is_viable.astype(int)
validate_y = validate.is_viable.astype(int)
test_y = test.is_viable.astype(int)


def copy_files(file_list, save_dir):
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for src_file in file_list:
        dest_file = os.path.join(save_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dest_file)
        print(f"已复制 {src_file} 到 {dest_file}")
src_files = ['Train_model_Liver4.py']  # 需要复制的文件列表

from keras.layers import Dense, LSTM 
from keras.initializers import GlorotUniform, HeNormal

def alternative_model00(L1=90, L2=16, dropout_rate=0.2, reg_coeff=1e-4):
    model = Sequential()
    # 第一层 LSTM
    model.add(LSTM(L1, 
                   return_sequences=True, 
                   input_shape=(7, 20), 
                   kernel_initializer=LecunNormal(),
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    
    # 第二层 LSTM
    model.add(LSTM(L2, 
                   return_sequences=False, 
                   kernel_initializer=Orthogonal(), 
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    
    # Dense + sigmoid
    model.add(Dense(units=1, 
                    activation='sigmoid',
                    kernel_initializer=RandomNormal(), 
                    kernel_regularizer=l2(reg_coeff)))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def alternative_model(L1=128, L2=16, dropout_rate=0.2, reg_coeff=1e-4):
    model = Sequential()
    # 第一层 LSTM 使用 LecunNormal 初始化 + L2 正则化 + Dropout
    model.add(LSTM(L1, 
                   return_sequences=True, 
                   input_shape=(7, 20), 
                   kernel_initializer=LecunNormal(),
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    
    # 第二层 LSTM 使用 Orthogonal 初始化 + L2 正则化 + Dropout
    model.add(LSTM(L2, 
                   return_sequences=False, 
                   kernel_initializer=Orthogonal(), 
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    
    # Dense 层 使用 RandomNormal 初始化 + L2 正则化
    model.add(Dense(units=1, 
                    kernel_initializer=RandomNormal(), 
                    kernel_regularizer=l2(reg_coeff)))
    
        # 修改 Dense 层，使用 sigmoid 激活函数
    model.add(Dense(units=1, 
                    activation='sigmoid',
                    kernel_initializer=RandomNormal(), 
                    kernel_regularizer=l2(reg_coeff)))
    # 修改损失函数和metrics
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

### 多显卡训练
pretrained_model_path = r'fit4function_models/Class_Liver_20240917_232339/epoch631_loss_0.3168_acc_0.8704_val_loss_0.3237_val_acc_0.8763.keras'
# 使用 MirroredStrategy 来并行化训练
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')
# 在策略作用域内创建模型
with strategy.scope():
    model = alternative_model()
    # 加载之前训练的模型
    # model = load_model(pretrained_model_path)

# 单显卡训练
# model = parent_model()

# Training parameters 
batch_size = 10000
EpochCount = 400
 
# Create a directory with the current time as its name
current_time = datetime.now().strftime('Class_Produc_%Y%m%d_%H%M%S')

save_directory = f'fit4function_models/{current_time}'
# copy_files(src_files, save_directory)

import sys
from keras.callbacks import Callback
class HideEpochHeader(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # 覆盖掉默认的 epoch 开头提示
        if hasattr(self.model, 'stop_training') and epoch >= 0:
            # 什么也不做，直接吞掉
            pass


from keras.callbacks import ReduceLROnPlateau
# 当验证集的损失不再改善时，将学习率减少10倍
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=80, min_lr=1e-6)


model.fit(train_x, train_y, batch_size=batch_size, epochs=EpochCount, 
          validation_data=(validate_x, validate_y), verbose=1,# 保持进度条
          callbacks=[HideEpochHeader(),
              CustomEarlyStopping(ratio=0.90, patience=500, restore_best_weights=True),
              SaveModelCallback_class(directory=save_directory),
              reduce_lr  # 动态调整学习率
          ])

# 预测验证集
validate_predictions = model.predict(validate_x)
validate_predictions = (validate_predictions > 0.5).astype(int)  # 使用阈值0.5来确定类别
# 计算准确率
accuracy_val = np.mean(validate_predictions.flatten() == validate_y)

# 预测测试集
test_predictions = model.predict(test_x)
test_predictions = (test_predictions > 0.5).astype(int)
# 计算准确率
accuracy_test = np.mean(test_predictions.flatten() == test_y)

# 打印准确率
print(f'Validation Accuracy: {accuracy_val:.4f}')
print(f'Test Accuracy: {accuracy_test:.4f}')


best_val_loss_path, best_val_mae_path = find_best_models2(save_directory)
print("Best val_loss model:", best_val_loss_path)
print("Best val_mae model:", best_val_mae_path)


# 保存 Pearson 相关性到文件
with open(os.path.join(save_directory, 'pearson_correlation.txt'), 'a') as f:
    f.write(f'Validation Accuracy: {accuracy_val:.4f}\n')
    f.write(f'Test Accuracy: {accuracy_test:.4f}\n')
    # f.write(f'best_val_loss_path : {best_val_loss_path}\n')
    # f.write(f'best_val_mae_path : {best_val_mae_path}\n')

'''这是用来公司的，为了规避，用了 qwen 进行改版'''