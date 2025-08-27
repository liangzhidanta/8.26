# 📁 项目结构说明

## 🎯 核心文件

### 训练相关
- **`train.sh`** - Shell训练脚本，支持超参数配置
- **`train_real_data.py`** - 主训练脚本，包含早停机制
- **`model.py`** - 低秩分解模型定义

### 数据处理
- **`qwenvl_dataloader.py`** - QwenVL数据加载器，支持特征缓存
- **`test_real_qwenvl.py`** - 数据加载器测试脚本

### 分析与可视化
- **`compare_svd_nn.py`** - SVD vs 神经网络性能比较
- **`visualize_svd_nn.py`** - 空间布局可视化（16×16特征图）
- **`svd_baseline.py`** - SVD基准实现

## 📊 数据目录

### 特征缓存
- **`feature_cache_real/`** - 4370个特征文件，每个256×2048维度
  - `features_0.pt` ~ `features_4369.pt`

### 训练结果
- **`results/`** - 训练生成的结果
  - `best_model.pt` - 最佳模型权重
  - `history.json` - 训练历史记录
  - `loss_curves.png` - 训练曲线
  - `error_heatmap.png` - 重构误差热力图

## 🔧 配置文件

- **`README.md`** - 完整的项目说明文档
- **环境要求**: conda mllm环境，PyTorch，CUDA支持

## 🚀 使用流程

1. **训练模型**: `./train.sh [参数]` 或 `python train_real_data.py [参数]`
2. **性能比较**: `python compare_svd_nn.py --rank 128`
3. **可视化结果**: `python visualize_svd_nn.py --image_idx 0 --rank 128`

## ✨ 特色功能

- **早停机制**: 自动停止过拟合
- **特征缓存**: 避免重复处理图像
- **空间可视化**: 16×16特征图展示前8个通道
- **超参数配置**: 支持命令行参数和Shell脚本
- **SVD基准**: 理论最优解对比

---

**项目状态**: ✅ 完全可用，所有功能已实现并测试通过
