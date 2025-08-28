#!/bin/bash

# 训练参数配置
ARCH="AdaLoRA"
EPOCHS=500
LEARNING_RATE="1e-4"
RANK=128
RESULTS_DIR="results"
DROPOUT=0.1
LAMBDA_ACT="1e-3"  # 新增的lambda_act参数
ALLOCATOR_INIT_WARMUP=1000
ALLOCATOR_FINAL_WARMUP=1000
ALLOCATOR_MASK_INTERVAL=20
ALLOCATOR_KEEP_RATIO=0.25

# 打印训练配置信息
echo "===== 训练配置 ====="
echo "架构: $ARCH"
echo "训练轮次: $EPOCHS"
echo "学习率: $LEARNING_RATE"
echo "Rank值: $RANK"
echo "结果保存目录: $RESULTS_DIR"
echo "dropout率: $DROPOUT"
echo "lambda_act参数: $LAMBDA_ACT"  # 显示新增参数
echo "使用rank分配器: 启用"
echo "分配器初始热身步数: $ALLOCATOR_INIT_WARMUP"
echo "分配器最终热身步数: $ALLOCATOR_FINAL_WARMUP"
echo "分配器掩码间隔: $ALLOCATOR_MASK_INTERVAL"
echo "分配器保留比例: $ALLOCATOR_KEEP_RATIO"
echo "===================="

# 创建结果目录（如果不存在）
mkdir -p $RESULTS_DIR

# 执行训练命令
python train_real_data.py \
  --arch $ARCH \
  --batch_size 128 \
  --epochs $EPOCHS \
  --lr $LEARNING_RATE \
  --rank $RANK \
  --results_dir $RESULTS_DIR \
  --use_rank_allocator \
  --allocator_init_warmup $ALLOCATOR_INIT_WARMUP \
  --allocator_final_warmup $ALLOCATOR_FINAL_WARMUP \
  --allocator_mask_interval $ALLOCATOR_MASK_INTERVAL \
  --allocator_keep_ratio $ALLOCATOR_KEEP_RATIO \
  --lambda_act $LAMBDA_ACT  # 添加新参数

# 检查训练是否成功
if [ $? -eq 0 ]; then
  echo "===== 训练完成 ====="
  echo "训练成功！结果已保存至: $RESULTS_DIR"
else
  echo "===== 训练失败 ====="
  echo "训练过程中出现错误，请检查日志信息"
  exit 1
fi
