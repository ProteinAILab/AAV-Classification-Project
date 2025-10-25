import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from keras.models import load_model

# 加载自定义回调函数和热编码函数
from utils_f4f import AA_hotencoding

# 读取保存的数据
output_folder = r'Liver/fit4function_library_screens'
train = pd.read_csv(os.path.join(output_folder, 'train.csv'), usecols=['AA', 'Output'])
validate = pd.read_csv(os.path.join(output_folder, 'validate.csv'), usecols=['AA', 'Output'])
test = pd.read_csv(os.path.join(output_folder, 'test.csv'), usecols=['AA', 'Output'])


# 热编码
AA_col = 'AA'
train_x =  np.asarray([AA_hotencoding(variant) for variant in train[AA_col]])
train_y = train.Output
validate_x = np.asarray([AA_hotencoding(variant) for variant in validate[AA_col]])
validate_y = validate.Output
test_x = np.asarray([AA_hotencoding(variant) for variant in test[AA_col]])
test_y = test.Output

# 加载训练好的模型
'''
Best val_loss model: Liver/fit4function_models/MAE_Liver_20240909_115454/epoch395_loss_0.6379_mae_0.5777_val_loss_0.6278_val_mae_0.5638.keras
Best val_mae model: Liver/fit4function_models/MAE_Liver_20240909_115454/epoch480_loss_0.6008_mae_0.5621_val_loss_0.6302_val_mae_0.5590.keras
Final model saved as fit4function_models/Class_Produc_20250921_230147/epoch400_loss_0.1735_acc_0.9405_val_loss_0.1933_val_acc_0.9317.keras
339/339 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step
170/170 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step
Validation Accuracy: 0.9321
Test Accuracy: 0.9347
Best val_loss model: fit4function_models/Class_Produc_20250921_230147/epoch400_loss_0.1735_acc_0.9405_val_loss_0.1933_val_acc_0.9317.keras
Best val_mae model: fit4function_models/Class_Produc_20250921_230147/epoch339_loss_0.1833_acc_0.9363_val_loss_0.1971_val_acc_0.9342.keras


'''
model_path = r'Liver/fit4function_models/MAE_Liver_20240909_115454/epoch480_loss_0.6008_mae_0.5621_val_loss_0.6302_val_mae_0.5590.keras'
# model_path = os.path.join(model_path, 'epoch477_loss_1.7105_mae_0.9881_val_loss_2.0208_val_mae_1.0891.keras')
model = load_model(model_path)

# 新建文件夹保存评估结果
current_time = datetime.now().strftime('Liver_Evaluation_%Y%m%d_%H%M%S')
save_directory = f'evaluation_results/{current_time}'
os.makedirs(save_directory, exist_ok=True)

log_transform = False  # False   True
# # 预测训练集
train_predictions = model.predict(train_x)
# 计算 Pearson 相关性 (Validation)
pearson_corr_train, _ = pearsonr(train_y, train_predictions.flatten())
print(f'Pearson Correlation (train): {pearson_corr_train:.4f}')

# # 预测验证集
validate_predictions = model.predict(validate_x)
if log_transform:  # 如果目标值是 log2 转换过的，那么预测值需要反向转换回原始尺度
    validate_predictions = 2 ** validate_predictions
# 计算 Pearson 相关性 (Validation)
pearson_corr_val, _ = pearsonr(validate_y, validate_predictions.flatten())
print(f'Pearson Correlation (Validation): {pearson_corr_val:.4f}')

# # 预测测试集
test_predictions = model.predict(test_x)
if log_transform:  # 如果 test_y 是 log2 转换过的
    test_predictions = 2 ** test_predictions
# 计算 Pearson 相关性 (Test)
pearson_corr_test, _ = pearsonr(test_y, test_predictions.flatten())
print(f'Pearson Correlation (Test): {pearson_corr_test:.4f}')


# 保存评估结果到文件
with open(os.path.join(save_directory, f'evaluation_results_{current_time}.txt'), 'w') as f:
    f.write(f'train Pearson Correlation: {pearson_corr_train:.4f}\n')
    f.write(f'Validation Pearson Correlation: {pearson_corr_val:.4f}\n')
    f.write(f'Test Pearson Correlation: {pearson_corr_test:.4f}\n')

print(f"评估结果已保存至: {save_directory}")


from Fitness_Scatter_Plot import plot_correlation  # 假设绘图函数存放在 plot_correlation.py 中
plot_correlation(train_y, train_predictions.flatten(), save_directory, f'train_correlation_plot_{current_time}')

# 生成验证集相关性图
plot_correlation(validate_y, validate_predictions.flatten(), save_directory, f'validation_correlation_plot_{current_time}')

# 生成测试集相关性图
plot_correlation(test_y, test_predictions.flatten(), save_directory, f'test_correlation_plot_{current_time}')


