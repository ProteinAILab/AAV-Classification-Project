# AAV分类训练项目

这是AAV（腺相关病毒）分类训练项目的代码库，用于训练和评估AAV9相关的分类模型。

## 项目文件说明

- `Production_clasify_aav9.py` - 主要的AAV9分类训练脚本
- `Production_clasify_aav9BK.py` - AAV9BK分类训练脚本
- `fit4function_library_screens/` - 适配函数库筛选相关文件
- `fit4function_models/` - 训练好的模型文件
- `utils_f4f.py` - 通用工具函数
- `model_inference_utils.py` - 模型推理工具
- `find_best_models.py` - 寻找最佳模型的脚本
- `Fitness_Scatter_Plot.py` - 适应度散点图绘制工具
- `Evaluate_Fit4Function_Model_Production.py` - 生产环境模型评估脚本
- `run_train.sh` - 训练启动脚本
- `stop_train.sh` - 训练停止脚本

## 使用方法

1. 运行训练：
   ```bash
   bash run_train.sh
   ```

2. 停止训练：
   ```bash
   bash stop_train.sh
   ```

3. 查看训练日志：
   ```bash
   tail -f Production_clasify_aav9.log
   ```

## 依赖项

- Python 3.x
- TensorFlow/PyTorch
- NumPy
- Pandas
- Matplotlib

## 注意事项

请确保有足够的计算资源（GPU推荐）来运行训练脚本。