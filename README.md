# 🎯 低秩分解训练系统（合并版 README）

基于 QwenVL3B 视觉特征的低秩自适应分解系统，支持多架构训练（ResMLP / MLP）、SVD基准对比与空间布局可视化，并提供早停与学习率调度等工程化能力。本 README 已合并先前分散在多个 .md 文件中的信息，去重且保持为最新版本。

## ✨ 功能总览

- **多架构**: `ResMLP`（残差多层感知机）与 `MLP`（纯线性层堆叠）可选
- **训练增强**: 早停、学习率调度（ReduceLROnPlateau）、恢复训练（--resume）
- **结果管理**: 不同架构的结果分目录保存：`results/<arch>/...`
- **比较与可视化**: SVD 对比、空间布局可视化（16×16前8通道）、误差热力图

## 🛠️ 环境准备

```bash
conda activate qwen
nvidia-smi
```

## 🚀 使用方式

### 训练（从头训练）
```bash
python train_real_data.py \
  --arch ResMLP \
  --epochs 500 \
  --lr 1e-4 \
  --rank 128 \
  --results_dir results
```

使用纯 MLP 架构：
```bash
python train_real_data.py \
  --arch MLP \
  --epochs 500 \
  --lr 1e-4 \
  --rank 128 \
  --results_dir results
```

使用 AdaLoRA 架构（动态秩 + 正交友好）：
```bash
python train_real_data.py \
  --arch AdaLoRA \
  --epochs 500 \
  --lr 1e-4 \
  --rank 128 \
  --results_dir results \
  --dropout 0.1 \
  --use_rank_allocator \
  --allocator_init_warmup 1000 \
  --allocator_final_warmup 1000 \
  --allocator_mask_interval 20 \
  --allocator_keep_ratio 0.25
# 说明：训练中会按上述超参自动更新掩码，实现动态秩控制
```

说明：产物将保存在 `results/<arch>/` 下，例如 `results/ResMLP/best_model.pt`。

### 恢复训练（基于当前架构的 best_model 继续）
```bash
python train_real_data.py \
  --arch ResMLP \
  --resume \
  --epochs 100 \
  --lr 1e-4 \
  --rank 128 \
  --results_dir results
```

说明：
- `--resume` 从 `results/<arch>/best_model.pt` 载入并从下一 epoch 继续。
- 本次的 `--epochs` 是追加训练的轮数（不是绝对 epoch 编号）。
- 学习率将记录到 `log/logs_时间戳/training.log`，调度器采用 `ReduceLROnPlateau` 按验证 MSE 自适应调整。

### 比较（SVD vs NN）
```bash
python compare_svd_nn.py \
  --arch ResMLP \
  --rank 128 \
  --results_dir results
```

对 AdaLoRA 训练的模型进行比较：
```bash
python compare_svd_nn.py \
  --arch AdaLoRA \
  --rank 128 \
  --results_dir results
```

### 可视化（单图对比）
```bash
python visualize_svd_nn.py \
  --arch ResMLP \
  --rank 128 \
  --results_dir results \
  --image_idx 0
```

可视化 AdaLoRA：
```bash
python visualize_svd_nn.py \
  --arch AdaLoRA \
  --rank 128 \
  --results_dir results \
  --image_idx 0
```

### 常见路径
- 最佳模型：`results/<arch>/best_model.pt`
- 训练历史：`results/<arch>/history.json`
- 曲线图：`results/<arch>/train_val_loss_comparison.png`

## 📋 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `ResMLP` | 模型架构（`ResMLP` 或 `MLP`） |
| `--batch_size` | 8 | 训练批次大小 |
| `--epochs` | 20 | 训练轮数或追加训练轮数（当 `--resume` 时） |
| `--lr` | 1e-4 | 学习率 |
| `--rank` | 128 | 低秩分解秩 |
| `--lambda_B` | 1e-4 | B矩阵正则化权重 |
| `--lambda_ortho` | 1e-3 | 正交性损失权重 |
| `--weight_decay` | 1e-5 | 权重衰减 |
| `--patience` | 10 | 早停耐心 |

## 📁 项目结构（简版）

```
low_rank_image/
├── train_real_data.py         # 训练（支持 --arch / --resume）
├── compare_svd_nn.py          # SVD vs NN 比较（按架构加载）
├── visualize_svd_nn.py        # 可视化（按架构加载）
├── model.py                   # 模型：LowRankProjSymmetric / LowRankProjMLP + 工厂方法
├── qwenvl_dataloader.py       # 数据加载 & 特征缓存
├── svd_baseline.py            # SVD基准
├── train.sh                   # 命令行训练脚本
├── feature_cache_real/        # 256×2048 特征缓存
└── results/
    ├── ResMLP/
    │   ├── best_model.pt
    │   ├── history.json
    │   └── train_val_loss_comparison.png
    └── MLP/
        └── ...
```

## 🔧 技术要点

- 输入/输出：`[batch, 256, 2048]`，重构 `X_hat ≈ A × B`
- 优化器：AdamW；调度器：ReduceLROnPlateau（基于验证 MSE）
- 训练日志：`log/logs_时间戳/`（包含 config、training、final_summary）
- 数据集：`textVQA_groundingtask_bbox`（特征维度 `[256, 2048]`，图像尺寸 448×448，patch 16×16）

### AdaLoRA 架构说明
- 模型名：`AdaptiveLowRankProjAdaLoRA`（工厂名：`AdaLoRA`/`adaptive_adalora`/`adaptive`）
- 结构：使用 `SVDLinear` 堆叠构建 `row_net` 与 `col_net`；引入全局可训练缩放 `lora_E ∈ R^{r×1}`，在前向中按秩通道缩放 `A_raw` 实现“可掩码的奇异值”；`effective_rank_mask()` 与 `current_effective_rank()` 便于外部 RankAllocator 集成。
- 正交正则：建议对 `SVDLinear` 的 `lora_A/lora_B` 做参数级正交；可选对 `A` 做激活级正交（`activation_orth_regu_A`）。

## 🔍 故障排除

1. CUDA 内存不足：减小 `--batch_size`
2. 不收敛：降低学习率或尝试 `ResMLP` 架构，增大 `--epochs`
3. 无法加载模型：确认对应架构的 `results/<arch>/best_model.pt` 已生成
4. 特征加载失败：检查 `feature_cache_real/` 是否完整

## 📄 许可证

本项目仅供学术研究使用。

---

提示：首次运行可能触发模型或数据缓存下载，请确保网络与磁盘空间充足。