#!/bin/bash

# 查找并杀掉 run_train.sh 启动的 Train_model_Liver4.py 进程
PIDS=$(ps -ef | grep "Production_clasify_aav9.py" | grep -v "grep" | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有找到 Production_clasify_aav9.py 相关进程。"
else
    echo "找到进程 PID: $PIDS"
    kill -9 $PIDS
    echo "已终结所有 Production_clasify_aav9.py 进程。"
fi
