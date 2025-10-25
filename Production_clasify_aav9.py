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
from keras.layers import Dense, LSTM, Dropout, Input
from keras.regularizers import l2
from keras.initializers import RandomNormal, LecunNormal, Orthogonal
from keras.models import load_model


def limit_gpu_memory(gb_limits, memory_limit_percent=None):
    """限制 GPU 内存的占位符函数。您需要实现或导入真实逻辑。"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                if gb_limits and i < len(gb_limits):
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gb_limits[i] * 1024)] # 将 GB 转换为 MB
                    )
        except RuntimeError as e:
            print(e)

def find_best_models2(directory):
    """查找目录中最佳模型文件的占位符函数。您需要实现或导入真实逻辑。"""
    # 这个函数应该在目录中查找最佳模型文件
    # 目前，它返回虚拟路径
    return os.path.join(directory, "dummy_best_loss.keras"), os.path.join(directory, "dummy_best_mae.keras")

def CustomEarlyStopping(ratio, patience, restore_best_weights):
    """自定义早停回调的占位符函数。您需要实现或导入真实逻辑。"""
    # 这应该是一个自定义的 Keras 回调类
    # 目前，我们使用一个占位符，它返回标准的 EarlyStopping
    from keras.callbacks import EarlyStopping
    return EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=restore_best_weights)

def SaveModelCallback_class(directory):
    """保存模型回调的占位符函数。您需要实现或导入真实逻辑。"""
    # 这应该是一个自定义的 Keras 回调类
    # 目前，我们使用一个占位符，它返回标准的 ModelCheckpoint
    from keras.callbacks import ModelCheckpoint
    filepath = os.path.join(directory, "model_{epoch:02d}_{val_loss:.4f}.keras")
    # 移除 save_format 参数
    return ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True) # 默认或通过扩展名推断格式

def amino_acid_one_hot_encode(protein_sequence):
    """将氨基酸序列编码为 one-hot 编码矩阵。"""
    # 这是一个占位符。实际编码逻辑取决于您的氨基酸字母表。
    # 标准 20 种氨基酸: A, R, N, D, C, E, Q, G, H, I, L, K, M, F, P, S, T, W, Y, V
    amino_acids = list("ARNDCQEGHILKMFPSTWYV") # 20 种标准氨基酸
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    max_len = 7 # 假设长度固定为 7，来自 input_shape
    one_hot = np.zeros((max_len, 20), dtype=np.float32)

    for i, aa in enumerate(protein_sequence):
        if i >= max_len:
            break
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1.0
        # else: 如有必要，处理未知氨基酸
    return one_hot

def prepare_data_from_csv(train_path, val_path, test_path, sequence_col, label_col):
    """从 CSV 文件加载并准备训练、验证和测试数据。"""
    train_df = pd.read_csv(train_path, usecols=[sequence_col, label_col])
    val_df = pd.read_csv(val_path, usecols=[sequence_col, label_col])
    test_df = pd.read_csv(test_path, usecols=[sequence_col, label_col])

    train_x = np.array([amino_acid_one_hot_encode(seq) for seq in train_df[sequence_col]])
    val_x = np.array([amino_acid_one_hot_encode(seq) for seq in val_df[sequence_col]])
    test_x = np.array([amino_acid_one_hot_encode(seq) for seq in test_df[sequence_col]])

    train_y = train_df[label_col].astype(int).values
    val_y = val_df[label_col].astype(int).values
    test_y = test_df[label_col].astype(int).values

    return train_x, train_y, val_x, val_y, test_x, test_y

def build_simple_lstm_model(l1_units=128, l2_units=16, dropout_rate=0.2, reg_coeff=1e-4):
    """构建模型的第一个 LSTM 层（包含 Input 层）。"""
    model = Sequential()
    # 添加 Input 层
    model.add(Input(shape=(7, 20)))
    model.add(LSTM(l1_units, 
                   return_sequences=True, 
                   kernel_initializer=LecunNormal(),
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    return model

def add_second_lstm_and_output_layers(model, l2_units=16, dropout_rate=0.2, reg_coeff=1e-4):
    """添加第二个 LSTM 层和最终的输出层。"""
    # 第二个 LSTM 层
    model.add(LSTM(l2_units, 
                   return_sequences=False, 
                   kernel_initializer=Orthogonal(), 
                   kernel_regularizer=l2(reg_coeff)))
    model.add(Dropout(dropout_rate))
    
    # 第一个 Dense 层（激活函数之前）
    model.add(Dense(units=1, 
                    kernel_initializer=RandomNormal(), 
                    kernel_regularizer=l2(reg_coeff)))
    
    # 第二个 Dense 层，使用 sigmoid 激活函数
    model.add(Dense(units=1, 
                    activation='sigmoid',
                    kernel_initializer=RandomNormal(), 
                    kernel_regularizer=l2(reg_coeff)))
    return model

def create_final_model(l1_units=128, l2_units=16, dropout_rate=0.2, reg_coeff=1e-4):
    """通过组合组件创建完整的模型。"""
    model = build_simple_lstm_model(l1_units, l2_units, dropout_rate, reg_coeff)
    model = add_second_lstm_and_output_layers(model, l2_units, dropout_rate, reg_coeff)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def calculate_binary_accuracy(predictions, true_labels):
    """计算二分类预测的准确率。"""
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes.flatten() == true_labels)
    return accuracy

# --- 主执行部分 ---
limit_gpu_memory(gb_limits=[2, 2, 2], memory_limit_percent=None)

# --- 数据加载和准备 ---
data_folder = r'./fit4function_library_screens'
train_file_path = os.path.join(data_folder, 'train_data.csv')
val_file_path = os.path.join(data_folder, 'val_data.csv')
test_file_path = os.path.join(data_folder, 'test_data.csv')

sequence_column_name = 'AA'
label_column_name = 'is_viable'

train_features, train_targets, val_features, val_targets, test_features, test_targets = prepare_data_from_csv(
    train_file_path, val_file_path, test_file_path, sequence_column_name, label_column_name
)

# 注意：此处移除了原始的 'model_architecture' 变量定义，
# 因为模型现在在策略作用域内创建。

# --- 多 GPU 策略 ---
pretrained_model_file_path = r'fit4function_models/Class_Liver_20240917_232339/epoch631_loss_0.3168_acc_0.8704_val_loss_0.3237_val_acc_0.8763.keras'
multi_gpu_strategy = tf.distribute.MirroredStrategy()
print(f'可用的设备数量: {multi_gpu_strategy.num_replicas_in_sync}')

with multi_gpu_strategy.scope():
    # 使用修正后的函数名和参数
    distributed_model = create_final_model(l1_units=128, l2_units=16, dropout_rate=0.2, reg_coeff=1e-4)
    # 如果加载预训练模型：
    # distributed_model = load_model(pretrained_model_file_path)

# --- 训练配置 ---
batch_size_for_training = 10000
number_of_epochs = 400

# 为本次训练运行创建一个唯一的目录
current_run_timestamp = datetime.now().strftime('Class_Produc_%Y%m%d_%H%M%S')
model_save_directory = f'fit4function_models/{current_run_timestamp}'

# 确保目录存在
os.makedirs(model_save_directory, exist_ok=True)

# --- 自定义回调 ---
class SuppressEpochBeginMessage(Callback):
    """一个抑制默认 epoch 开始消息的回调。"""
    def on_epoch_begin(self, epoch, logs=None):
        pass  # 什么都不做以抑制输出

from keras.callbacks import ReduceLROnPlateau
learning_rate_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=80, min_lr=1e-6)

# --- 训练执行 ---
distributed_model.fit(
    train_features, train_targets,
    batch_size=batch_size_for_training,
    epochs=number_of_epochs,
    validation_data=(val_features, val_targets),
    verbose=1,  # 保留进度条
    callbacks=[
        SuppressEpochBeginMessage(),
        CustomEarlyStopping(ratio=0.90, patience=500, restore_best_weights=True),
        SaveModelCallback_class(directory=model_save_directory),
        learning_rate_reducer  # 动态调整学习率
    ]
)

# --- 模型评估 ---
val_predictions_raw = distributed_model.predict(val_features)
val_accuracy_score = calculate_binary_accuracy(val_predictions_raw, val_targets)

test_predictions_raw = distributed_model.predict(test_features)
test_accuracy_score = calculate_binary_accuracy(test_predictions_raw, test_targets)

print(f'验证集准确率分数: {val_accuracy_score:.4f}')
print(f'测试集准确率分数: {test_accuracy_score:.4f}')

# --- 查找最佳模型 ---
best_val_loss_model_path, best_val_mae_model_path = find_best_models2(model_save_directory)
print("最佳验证损失模型路径:", best_val_loss_model_path)
print("最佳验证 MAE 模型路径:", best_val_mae_model_path)

# --- 保存结果 ---
results_log_file_path = os.path.join(model_save_directory, 'evaluation_results.txt')
with open(results_log_file_path, 'a') as log_file:
    log_file.write(f'验证集准确率分数: {val_accuracy_score:.4f}\n')
    log_file.write(f'测试集准确率分数: {test_accuracy_score:.4f}\n')
    # log_file.write(f'最佳验证损失模型路径: {best_val_loss_model_path}\n')
    # log_file.write(f'最佳验证 MAE 模型路径: {best_val_mae_model_path}\n')
