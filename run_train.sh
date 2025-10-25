#!/bin/bash

# 切换到脚本所在目录（可选）
cd "$(dirname "$0")"

# 使用 nohup 后台运行 Python 脚本
TF_CPP_MIN_LOG_LEVEL=2 nohup python3 Production_clasify_aav9.py > Production_clasify_aav9.log 2>&1 &

echo "Production_clasify_aav9.py 已启动，日志保存在 Production_clasify_aav9.log"
