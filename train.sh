#!/bin/bash

# 训练脚本 - 使用真实QwenVL数据的低秩分解模型

# 默认参数
BATCH_SIZE=${1:-64}
EPOCHS=${2:-200}
LR=${3:-0.0001}
RANK=${4:-128}
LAMBDA_B=${5:-0.0001}
LAMBDA_ORTHO=${6:-0.001}
RESULTS_DIR=${7:-"results"}

echo "🚀 开始训练低秩分解模型..."
echo "=================================="
echo "超参数配置:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Rank: $RANK"
echo "  Lambda B: $LAMBDA_B"
echo "  Lambda Ortho: $LAMBDA_ORTHO"
echo "  结果目录: $RESULTS_DIR"
echo "=================================="

# 检查GPU可用性
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 检查GPU状态..."
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU训练"
fi

# 激活conda环境
echo "🔧 激活conda环境..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mllm

# 开始训练
echo "🎯 开始训练..."
python train_real_data.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --rank $RANK \
    --lambda_B $LAMBDA_B \
    --lambda_ortho $LAMBDA_ORTHO \
    --results_dir $RESULTS_DIR

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "📊 结果保存在: $RESULTS_DIR/"
    
    # 显示结果文件
    if [ -d "$RESULTS_DIR" ]; then
        echo "📁 生成的文件:"
        ls -la "$RESULTS_DIR/"
    fi
else
    echo "❌ 训练失败！"
    exit 1
fi

echo "🎉 训练脚本执行完成！"
