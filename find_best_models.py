
import os
import re

def find_best_models(folder_path):
    # 用于存储文件信息的列表
    models = []

    # 正则表达式，匹配文件名中的 loss 和 mae 值
    pattern = r'epoch\d+_loss_(\d+\.\d+)_mae_(\d+\.\d+)_val_loss_(\d+\.\d+)_val_mae_(\d+\.\d+)\.keras'
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 使用正则表达式匹配文件名
        match = re.match(pattern, filename)
        if match:
            loss, mae, val_loss, val_mae = map(float, match.groups())
            models.append({
                'filename': filename,
                'val_loss': val_loss,
                'val_mae': val_mae
            })
    
    # 找到 val_loss 最小的文件
    best_val_loss_model = min(models, key=lambda x: x['val_loss'])
    
    # 找到 val_mae 最小的文件
    best_val_mae_model = min(models, key=lambda x: x['val_mae'])
    
    # 返回两个最优模型的文件路径
    return (os.path.join(folder_path, best_val_loss_model['filename']),
            os.path.join(folder_path, best_val_mae_model['filename']))

import re
import os

def find_best_models2(folder_path):
    # 用于存储文件信息的列表
    models = []

    # 正则表达式，匹配文件名中的 loss 和 accuracy 值
    pattern = r'epoch\d+_loss_(\d+\.\d+)_acc_(\d+\.\d+)_val_loss_(\d+\.\d+)_val_acc_(\d+\.\d+)\.keras'
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 使用正则表达式匹配文件名
        match = re.match(pattern, filename)
        if match:
            loss, acc, val_loss, val_acc = map(float, match.groups())
            models.append({
                'filename': filename,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
    
    # 找到 val_loss 最小的文件
    best_val_loss_model = min(models, key=lambda x: x['val_loss'])
    
    # 找到 val_acc 最大的文件（因为准确率越高越好）
    best_val_acc_model = max(models, key=lambda x: x['val_acc'])
    
    # 返回两个最优模型的文件路径
    return (os.path.join(folder_path, best_val_loss_model['filename']),
            os.path.join(folder_path, best_val_acc_model['filename']))

if __name__ == "__main__":
    # 使用该函数
    folder_path = 'fit4function_models/Class_Produc_20240918_121823'  # 替换为你的文件夹路径
    best_val_loss_path, best_val_mae_path = find_best_models2(folder_path)

    print("Best val_loss model:", best_val_loss_path)
    print("Best val_mae model:", best_val_mae_path)