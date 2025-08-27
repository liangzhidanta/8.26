# 🎯 低秩分解训练系统

基于QwenVL3B视觉特征的低秩自适应分解训练系统，支持SVD基准对比和空间布局可视化。

## ✨ 核心功能

### 🚀 训练系统
- **超参数配置**: 支持命令行参数和Shell脚本
- **早停机制**: 连续10个验证损失不下降自动停止
- **特征缓存**: 避免重复处理图像，提高效率

### 🔍 性能对比
- **SVD基准**: 截断SVD理论最优解
- **神经网络**: 低秩自适应分解模型
- **定量评估**: 余弦相似度、重构误差等指标

### 🎨 可视化系统
- **空间布局**: 16×16特征图展示前8个通道
- **对比分析**: 原始特征 vs SVD重构 vs 神经网络重构
- **误差分布**: 重构误差的空间分布可视化

## 🛠️ 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate mllm

# 检查GPU
nvidia-smi
```

### 2. 训练模型
```bash
# 使用Shell脚本（推荐）
./train.sh [batch_size] [epochs] [lr] [rank] [lambda_B] [lambda_ortho] [results_dir]

# 示例：基础训练
./train.sh 8 20 0.001 128 0.0001 0.001 results

# 直接使用Python
python train_real_data.py --batch_size 8 --epochs 20 --lr 0.001 --rank 128
```

### 3. 性能对比
```bash
# 比较SVD和神经网络
python compare_svd_nn.py --rank 128
```

### 4. 可视化结果
```bash
# 空间布局可视化
python visualize_svd_nn.py --image_idx 0 --rank 128 --results_dir results
```

## 📋 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 训练批次大小 |
| `--epochs` | 20 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--rank` | 128 | 低秩分解的秩 |
| `--lambda_B` | 0.0001 | B矩阵正则化权重 |
| `--lambda_ortho` | 0.001 | 正交性损失权重 |
| `--weight_decay` | 0.00001 | 权重衰减 |
| `--patience` | 10 | 早停耐心值 |

## 📁 项目结构

```
low_rank_image/
├── README.md                   # 项目说明文档
├── train.sh                    # Shell训练脚本
├── train_real_data.py         # 主训练脚本
├── compare_svd_nn.py          # SVD vs NN比较
├── visualize_svd_nn.py        # 空间布局可视化
├── qwenvl_dataloader.py       # 数据加载器
├── model.py                   # 低秩分解模型
├── svd_baseline.py            # SVD基准实现
├── test_real_qwenvl.py        # 数据加载器测试
├── feature_cache_real/        # 特征缓存目录
│   ├── features_0.pt         # 4370个特征文件
│   ├── features_1.pt
│   └── ...
└── results/                   # 训练结果目录
    ├── best_model.pt         # 最佳模型
    ├── history.json          # 训练历史
    ├── loss_curves.png       # 损失曲线
    └── ...
```

## 🔧 技术细节

### 特征提取
- **模型**: Qwen/Qwen2.5-VL-3B-Instruct
- **图像尺寸**: 448×448像素
- **Patch尺寸**: 16×16
- **特征维度**: [256, 2048] (16×16 tokens, 2048维特征)
- **数据集**: textVQA_groundingtask_bbox (4370个样本)

### 模型架构
- **输入**: [batch_size, 256, 2048]
- **输出**: [batch_size, 256, 2048]
- **分解**: X ≈ A × B，其中A∈[batch_size, 256, rank], B∈[batch_size, rank, 2048]

### 训练策略
- **优化器**: AdamW
- **学习率调度**: ReduceLROnPlateau
- **损失函数**: 重构损失 + B矩阵正则化 + 正交性损失
- **早停**: 验证MSE连续10个epoch不下降

## 📊 性能结果

### 测试集性能 (80个样本)
- **SVD平均余弦相似度**: 0.9919 ± 0.0021
- **NN平均余弦相似度**: 0.7935 ± 0.0371
- **性能差距**: SVD比神经网络高约20%

### 单图重构质量
- **SVD重构相似度**: 0.9910
- **NN重构相似度**: 0.8073

## 🚀 使用示例

### 完整训练流程
```bash
# 1. 开始训练
./train.sh 8 20 0.001 128 0.0001 0.001 results_custom

# 2. 等待训练完成（支持早停）

# 3. 比较性能
python compare_svd_nn.py --rank 128

# 4. 可视化结果
python visualize_svd_nn.py --image_idx 0 --rank 128 --results_dir results_custom
```

### 参数调优
```bash
# 尝试不同的rank值
./train.sh 8 20 0.001 64 0.0001 0.001 results_rank64
./train.sh 8 20 0.001 256 0.0001 0.001 results_rank256

# 尝试不同的学习率
./train.sh 8 20 0.0005 128 0.0001 0.001 results_lr5e4
./train.sh 8 20 0.002 128 0.0001 0.001 results_lr2e3
```

## 🔍 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch_size
2. **训练不收敛**: 调整学习率或lambda参数
3. **特征加载失败**: 检查feature_cache_real目录
4. **模型文件缺失**: 先运行训练脚本

### 调试模式
```bash
# 测试数据加载器
python test_real_qwenvl.py

# 检查特征维度
ls -la feature_cache_real/ | head -5
```

## 📈 扩展功能

### 可能的改进
1. **多GPU训练**: 支持分布式训练
2. **更多可视化**: 交互式可视化、更多通道展示
3. **超参数优化**: 自动超参数搜索
4. **模型压缩**: 量化、剪枝等优化

## 📄 许可证

本项目仅供学术研究使用。

---

**注意**: 首次运行会自动下载QwenVL3B模型（约6GB），请确保网络连接正常。



