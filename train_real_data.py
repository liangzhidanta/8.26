#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低秩分解训练脚本 - 使用真实QwenVL数据
支持早停机制和超参数配置
"""

import os
import argparse
os.environ["MODELSCOPE_CACHE"] = "/data2/dzr/.cache" 
os.environ["TOKENS_PARALLELISM"] = "false"
import math
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import shutil

from qwenvl_dataloader import RealQwenVLDataset, create_data_loaders_real
from model import LowRankProjSymmetric, ResidualMLP
from svd_baseline import svd_truncated_error


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def layernorm_along_feature(X: torch.Tensor) -> torch.Tensor:
    """沿着特征维度进行层归一化"""
    return torch.nn.functional.layer_norm(X, normalized_shape=(X.size(-1),))


def train_one_epoch(model, dataloader, optimizer, device, lambda_B, lambda_ortho):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    first_batch = True
    
    # 添加进度条
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for (X,) in pbar:
        if first_batch:
            print(f"🔍 DEBUG: Input tensor X shape (before transfer): {X.shape}")
            print(f"🔍 DEBUG: Input tensor X dtype (before transfer): {X.dtype}")
            print(f"🔍 DEBUG: Input tensor X device (before transfer): {X.device}")
            print(f"🔍 DEBUG: Input tensor X min/max (before transfer): {X.min().item():.4f}/{X.max().item():.4f}")
            print(f"🔍 DEBUG: Input tensor X mean/std (before transfer): {X.mean().item():.4f}/{X.std().item():.4f}")
            first_batch = False
            
        # 明确地将数据移动到GPU
        X = X.to(device, non_blocking=True)
        if n_batches == 0:
            print(f"🔍 DEBUG: Input tensor X device (after transfer): {X.device}")
        
        X = layernorm_along_feature(X)
        
        if n_batches == 0:
            print(f"🔍 DEBUG: After layernorm X shape: {X.shape}")
            print(f"🔍 DEBUG: After layernorm X device: {X.device}")
            print(f"🔍 DEBUG: After layernorm X min/max: {X.min().item():.4f}/{X.max().item():.4f}")
            print(f"🔍 DEBUG: After layernorm X mean/std: {X.mean().item():.4f}/{X.std().item():.4f}")
        
        optimizer.zero_grad(set_to_none=True)
        X_hat, A, B = model(X)
        
        if n_batches == 0:
            print(f"🔍 DEBUG: Model output X_hat shape: {X_hat.shape}")
            print(f"🔍 DEBUG: Model output A shape: {A.shape}")
            print(f"🔍 DEBUG: Model output B shape: {B.shape}")
        
        # 仅使用MSE损失进行优化
        mse = torch.mean((X - X_hat) ** 2)
        loss = mse
        
        if n_batches == 0:
            print(f"🔍 DEBUG: MSE (reconstruction error): {mse.item():.6f}")
            print(f"🔍 DEBUG: Total loss (MSE only): {loss.item():.6f}")
        
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        n_batches += 1
        
        # 更新进度条显示当前loss
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse.item():.4f}'
        })
    
    pbar.close()
    return {
        "loss": total_loss / max(n_batches, 1),
        "mse": total_mse / max(n_batches, 1),
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_mse = 0.0
    total_rel_f = 0.0
    total_evr = 0.0
    n_batches = 0
    
    # 添加进度条
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for (X,) in pbar:
            # 明确地将数据移动到GPU
            X = X.to(device, non_blocking=True)
            X = layernorm_along_feature(X)
            X_hat, A, B = model(X)
            
            # MSE
            mse = torch.mean((X - X_hat) ** 2)
            total_mse += mse.item()
            
            # Rel_F
            rel_f = torch.mean(torch.norm(X - X_hat, dim=-1) / (torch.norm(X, dim=-1) + 1e-8))
            total_rel_f += rel_f.item()
            
            # EVR
            evr = torch.mean(torch.norm(X_hat, dim=-1) / (torch.norm(X, dim=-1) + 1e-8))
            total_evr += evr.item()
            
            n_batches += 1
            
            # 更新进度条显示当前指标
            pbar.set_postfix({
                'mse': f'{mse.item():.4f}',
                'rel_f': f'{rel_f.item():.4f}',
                'evr': f'{evr.item():.4f}'
            })
    
    pbar.close()
    return {
        "mse": total_mse / max(n_batches, 1),
        "rel_f": total_rel_f / max(n_batches, 1),
        "evr": total_evr / max(n_batches, 1),
    }


def save_history_json(history, path):
    """保存训练历史为JSON"""
    import json
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def plot_curves(history, results_dir):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt
    
    epochs = [log['epoch'] for log in history]
    train_losses = [log['train_loss'] for log in history]
    val_mses = [log['val_mse'] for log in history]
    rel_f_scores = [log['val_rel_f'] for log in history]
    evr_scores = [log['val_evr'] for log in history]
    
    # 损失曲线
    plt.figure(figsize=(15, 5))
    
    # 第一个子图：训练损失和验证损失在同一张图上
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    plt.plot(epochs, val_mses, 'r-', linewidth=2, label='Val MSE', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第二个子图：验证MSE（保持原样）
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_mses, 'r-', linewidth=2, label='Val MSE', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第三个子图：Rel_F和EVR（保持原样）
    plt.subplot(1, 3, 3)
    plt.plot(epochs, rel_f_scores, 'g-', linewidth=2, label='Rel_F', marker='^', markersize=4)
    plt.plot(epochs, evr_scores, 'm-', linewidth=2, label='EVR', marker='d', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Rel_F and EVR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 单独的曲线
    # 训练损失和验证损失对比图
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b-', linewidth=3, label='Train Loss', marker='o', markersize=6)
    plt.plot(epochs, val_mses, 'r-', linewidth=3, label='Val MSE', marker='s', markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training vs Validation Loss Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加一些统计信息
    train_final = train_losses[-1] if train_losses else 0
    val_final = val_mses[-1] if val_mses else 0
    plt.text(0.02, 0.98, f'Final Train Loss: {train_final:.6f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.92, f'Final Val MSE: {val_final:.6f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'train_val_loss_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Rel_F曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rel_f_scores, 'g-', linewidth=2, label='Rel_F', marker='^', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Rel_F Score')
    plt.title('Relative F-norm Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'relF_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # EVR曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, evr_scores, 'm-', linewidth=2, label='EVR', marker='d', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('EVR Score')
    plt.title('Eigenvalue Ratio Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'evr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_error_heatmap(error_matrix, path):
    """保存误差热力图"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(error_matrix.cpu().numpy(), cmap='Reds', aspect='auto')
    plt.colorbar(label='Error Magnitude')
    plt.title('Reconstruction Error Heatmap')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Token Position')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_B_row_norm_hist(B_matrix, path):
    """保存B矩阵行范数分布直方图"""
    import matplotlib.pyplot as plt
    
    B_norms = torch.norm(B_matrix.squeeze(0), dim=1).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(B_norms, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Row Norm')
    plt.ylabel('Frequency')
    plt.title('B Matrix Row Norm Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train low-rank decomposition model with real QwenVL data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--rank', type=int, default=128, help='Rank for low-rank decomposition')
    parser.add_argument('--lambda_B', type=float, default=1e-4, help='Weight for B regularization')
    parser.add_argument('--lambda_ortho', type=float, default=1e-3, help='Weight for orthogonality loss')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    # 强制使用GPU设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU available, using CPU")

    # 超参数
    T = 256  # 256 tokens for 16x16 spatial layout
    D = 2048  # 2048 hidden dimensions
    r = args.rank
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    weight_decay = args.weight_decay
    lambda_B = args.lambda_B
    lambda_ortho = args.lambda_ortho
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # 打印配置信息
    print(f"Configuration:")
    print(f"  Image size: 448x448 (16x16 patches)")
    print(f"  Token count: {T}")
    print(f"  Feature dimension: {D}")
    print(f"  Rank: {r}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    # 加载真实数据集（使用真实的QwenVL3B模型）
    print("\nLoading textVQA dataset with real QwenVL3B model...")
    try:
        dataset = RealQwenVLDataset(
            dataset_dir="/data2/dzr/textVQA_groundingtask_bbox",
            image_size=448,
            use_cache=True,
            cache_dir="./feature_cache_real"
        )
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")
        
        # 检查特征矩阵维度
        print(f"🔍 Dataset features shape: {dataset.features.shape}")
        print(f"🔍 Dataset features dtype: {dataset.features.dtype}")
        print(f"🔍 Dataset features device: {dataset.features.device}")
        
        # 验证特征矩阵维度
        expected_shape = (len(dataset), 256, 2048)
        if dataset.features.shape != expected_shape:
            print(f"⚠️  WARNING: Unexpected feature shape!")
            print(f"   Expected: {expected_shape}")
            print(f"   Actual: {dataset.features.shape}")
            raise ValueError(f"Feature matrix shape mismatch: {dataset.features.shape} != {expected_shape}")
        else:
            print(f"✅ Feature matrix shape validation passed!")
        
        # 创建数据加载器 (80%, 10%, 10% 划分)
        train_loader, val_loader, test_loader = create_data_loaders_real(
            dataset, 
            batch_size=batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=4
        )
        print(f"✅ Data loaders created successfully")
        
        # 测试第一个batch
        print("\n🔍 Testing first batch from train loader...")
        for batch in train_loader:
            X = batch[0]
            print(f"🔍 First batch X shape: {X.shape}")
            print(f"🔍 First batch X dtype: {X.dtype}")
            print(f"🔍 First batch X device: {X.device}")
            break
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Falling back to dummy dataset...")
        
        # Fallback: 使用模拟数据
        from train import generate_dummy_dataset
        dataset = generate_dummy_dataset(2000, T, D)
        val_size = int(len(dataset) * 0.1)
        test_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"✅ Dummy dataset created with {len(dataset)} samples")

    # 创建模型
    model = LowRankProjSymmetric(input_dim=D, num_tokens=T, rank=r, normalize=False)
    model.to(device)
    print(f"✅ Model created and moved to {device}")

    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = math.inf
    best_path = os.path.join(results_dir, "best_model.pt")
    history = []
    
    # 早停机制
    patience_counter = 0
    best_epoch = 0

    print(f"\nStart training with early stopping (patience: {args.patience})...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, lambda_B, lambda_ortho)
        val_stats = evaluate(model, val_loader, device)
        scheduler.step(val_stats["mse"])
        epoch_time = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mse": train_stats["mse"],
            "val_mse": val_stats["mse"],
            "val_rel_f": val_stats["rel_f"],
            "val_evr": val_stats["evr"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
        }
        history.append(log)
        print(f"Epoch {epoch:03d} | train_loss={log['train_loss']:.6f} train_mse={log['train_mse']:.6f} "
              f"val_mse={log['val_mse']:.6f} rel_f={log['val_rel_f']:.6f} evr={log['val_evr']:.6f} "
              f"lr={log['lr']:.2e} time={epoch_time:.1f}s")

        # checkpoint
        if val_stats["mse"] < best_val:
            best_val = val_stats["mse"]
            best_epoch = epoch
            patience_counter = 0  # 重置耐心计数器
            # 保存带有epoch号的权重文件
            epoch_best_path = os.path.join(results_dir, f"best_model_epoch_{epoch:03d}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "T": T, "D": D, "rank": r,
                    "lambda_B": lambda_B, "lambda_ortho": lambda_ortho,
                    "image_size": 448, "patch_size": 16,
                    "dataset": "textVQA_groundingtask_bbox",
                    "data_split": "80-10-10",
                    "feature_extractor": "QwenVL3B_real",
                },
                "history": history,
                "epoch": epoch,
                "best_val_mse": best_val,
            }, epoch_best_path)
            # 同步更新一个固定名称的软链接/副本
            try:
                # 优先尝试符号链接更新
                if os.path.islink(best_path) or os.path.exists(best_path):
                    os.remove(best_path)
                os.symlink(os.path.basename(epoch_best_path), best_path)
            except Exception:
                # 若符号链接不可用，复制一份
                shutil.copyfile(epoch_best_path, best_path)
            print(f"✅ New best checkpoint saved at epoch {epoch}: {os.path.basename(epoch_best_path)} (Val MSE: {best_val:.6f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epochs (best: {best_val:.6f} at epoch {best_epoch})")
            
        # 早停检查
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"Best validation MSE: {best_val:.6f} at epoch {best_epoch}")
            break

    # 验证集上抽样SVD下界评估
    if len(val_loader.dataset) > 0:
        print("\nEvaluating SVD lower bound...")
        sample_loader = DataLoader(val_loader.dataset, batch_size=1, shuffle=True)
        default_samples = min(10, len(val_loader.dataset))
        svd_samples = int(os.environ.get("SVD_SAMPLES", default_samples))
        errs = []
        count = 0
        with torch.no_grad():
            for (X,) in sample_loader:
                X = layernorm_along_feature(X)  # 与模型一致的预处理
                e = svd_truncated_error(X.squeeze(0), rank=r)
                errs.append(e)
                count += 1
                if count >= svd_samples:
                    break
        if errs:
            avg_svd_err = sum(errs) / len(errs)
            print(f"SVD lower-bound relative error (avg over {len(errs)} samples): {avg_svd_err:.6f}")

    # 测试集评估
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_stats = evaluate(model, test_loader, device)
        print(f"Test set - MSE: {test_stats['mse']:.6f}, Rel_F: {test_stats['rel_f']:.6f}, EVR: {test_stats['evr']:.6f}")
        
        # 保存测试结果
        test_results = {
            "test_mse": test_stats["mse"],
            "test_rel_f": test_stats["rel_f"],
            "test_evr": test_stats["evr"],
        }
        import json
        with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

    # 保存曲线与history
    save_history_json(history, os.path.join(results_dir, 'history.json'))
    if history:
        plot_curves(history, results_dir)

    # 误差热力图与B行范数分布（取一小批次样本）
    print("\nGenerating visualizations...")
    with torch.no_grad():
        sample_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=True)
        for (X,) in sample_loader:
            X = layernorm_along_feature(X.to(device))
            X_hat, A, B = model(X)
            err_mat = (X - X_hat).squeeze(0)
            save_error_heatmap(err_mat, os.path.join(results_dir, 'error_heatmap.png'))
            # B: (1, r, D) -> 取第一个样本
            save_B_row_norm_hist(B, os.path.join(results_dir, 'B_row_norm_hist.png'))
            break

    print(f"\n✅ Training completed!")
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
