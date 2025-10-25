
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def predict_sequence(model, encoded_sequences):
    predictions = model.predict(encoded_sequences)
    return predictions

def load_keras_models(model_paths):
    """
    加载多个Keras模型。
    :param model_paths: 模型路径的列表
    :return: 已加载的模型列表
    """
    models = [load_model(path) for path in model_paths]
    return models

##
def load_keras_model(model_path, gb_limit=None, memory_limit_percent=None):
    # 在加载模型之前应用GPU显存限制
    limit_gpu_memory(gb_limit, memory_limit_percent)
    return load_model(model_path)

def load_keras_models(model_paths, gb_limit=None, memory_limit_percent=None):
    """
    加载多个Keras模型。
    :param model_paths: 模型路径的列表
    :return: 已加载的模型列表
    """
    # 在加载模型之前应用GPU显存限制
    limit_gpu_memory(gb_limit, memory_limit_percent)
    models = [load_model(path) for path in model_paths]
    return models

def predict_sequences(models, encoded_sequences):
    predictions = [model.predict(encoded_sequences) for model in models]
    return predictions

##
def predict_sequences(models, encoded_sequences):
    """
    使用多个模型进行推理，并计算模型预测的平均值作为最终预测。
    models: 加载的模型列表
    encoded_sequences: 编码后的氨基酸序列
    return: 最终的集成预测结果
    """
    predictions = np.zeros((encoded_sequences.shape[0], 1))
    # 累加所有模型的预测结果
    for model in models:
        model_predictions = model.predict(encoded_sequences)
        predictions += model_predictions
    # 计算平均值
    predictions /= len(models)
    return predictions


def limit_gpu_memory_for_specific_gpu(gpu_id, gb_limit=None, memory_limit_percent=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('gpus : ',gpus)
    # gpus = [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU')]
    if gpus and gpu_id < len(gpus):
        try:
            gpu = gpus[gpu_id]
            if gb_limit is not None:
                print('gpu_id : ',gpu_id)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gb_limit * 1024)]
                )
            elif memory_limit_percent is not None:
                total_memory = 16 * 1024  # 假设总显存为16GB，可以根据实际情况调整
                memory_limit = int(total_memory * memory_limit_percent)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
        except RuntimeError as e:
            print(e)
            
            
import tensorflow as tf


def limit_gpu_memory0(gb_limit=None, memory_limit_percent=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if gb_limit is not None:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gb_limit * 1024)]
                    )
                elif memory_limit_percent is not None:
                    # 例如，假设总显存为16GB，可以根据实际情况调整
                    total_memory = 16 * 1024  # 获取GPU的总内存（以MB为单位）
                    memory_limit = int(total_memory * memory_limit_percent)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
        except RuntimeError as e:
            print(e)

            
def limit_gpu_memory(gb_limits=None, memory_limit_percent=None):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            virtual_devices = []
            for i, gpu in enumerate(gpus):
                if gb_limits is not None and i < len(gb_limits):
                    memory_limit = gb_limits[i] * 1024  # 将GB转换为MB
                elif memory_limit_percent is not None:
                    total_memory = 24564  # 设定为每块GPU的总内存，单位MB，可以根据实际情况调整
                    memory_limit = int(total_memory * memory_limit_percent)
                else:
                    memory_limit = None
                if memory_limit is not None:
                    virtual_devices.append(tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit))
                else:
                    virtual_devices.append(tf.config.experimental.VirtualDeviceConfiguration())
                
            for gpu, config in zip(gpus, virtual_devices):
                tf.config.experimental.set_virtual_device_configuration(gpu, [config])

        except RuntimeError as e:
            print(e)


        