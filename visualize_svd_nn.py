#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化SVD和神经网络低秩分解的比较结果
参考visualize_comparison.py的方式，使用空间布局展示前8个通道
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse

from qwenvl_dataloader import RealQwenVLDataset, create_data_loaders_real
from model import LowRankProjSymmetric
from svd_baseline import svd_truncated_error, svd_truncated_reconstruction


def reshape_patches_to_spatial(X: torch.Tensor, image_size: int = 448, patch_size: int = 16) -> torch.Tensor:
    """
    将patch序列重新排列为空间布局
    X: (T, D) -> (H, W, D) 其中 H*W = T
    对于448x448图像，16x16 patch，应该有256个token，对应16x16的空间布局
    """
    T, D = X.shape
    
    # 计算空间尺寸
    if T == 256:
        # 256 = 16*16，对应16x16的空间布局
        H, W = 16, 16
    elif T == 784:
        # 784 = 28*28，对应28x28的空间布局（448/16 = 28）
        H, W = 28, 28
    elif T == 196:
        # 196 = 14*14
        H, W = 14, 14
    else:
        # 尝试找到最接近的平方数
        side = int(np.sqrt(T))
        H, W = side, side
    
    # 如果T不等于H*W，进行padding或truncation
    if T < H * W:
        # 需要padding
        pad_size = H * W - T
        X_padded = torch.cat([X, torch.zeros(pad_size, D, device=X.device, dtype=X.dtype)], dim=0)
        return X_padded.view(H, W, D)
    elif T > H * W:
        # 需要truncation
        return X[:H*W].view(H, W, D)
    else:
        return X.view(H, W, D)


def visualize_channels_spatial(X: torch.Tensor, X_hat: torch.Tensor, X_svd: torch.Tensor, 
                              out_dir: str, image_idx: int = 0, image_size: int = 448, patch_size: int = 16) -> None:
    """可视化前8个channel的特征矩阵，每个channel显示为16x16的空间布局"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 取前8个channel
    num_channels = min(8, X.shape[-1])
    
    # 重塑为空间布局
    X_spatial = reshape_patches_to_spatial(X, image_size, patch_size)  # (H, W, D)
    X_hat_spatial = reshape_patches_to_spatial(X_hat, image_size, patch_size)
    X_svd_spatial = reshape_patches_to_spatial(X_svd, image_size, patch_size)
    
    # 创建3行8列的子图布局
    fig, axes = plt.subplots(3, num_channels, figsize=(20, 12))
    
    # 为每个channel创建可视化
    for ch in range(num_channels):
        # 原特征矩阵
        im1 = axes[0, ch].imshow(X_spatial[:, :, ch].detach().cpu().numpy(), 
                                 aspect='equal', cmap='viridis')  # 改为正方形展示
        axes[0, ch].set_title(f'Ch{ch+1}', fontsize=12)
        axes[0, ch].set_ylabel('Original', fontsize=12, fontweight='bold')
        
        # 神经网络重构
        im2 = axes[1, ch].imshow(X_hat_spatial[:, :, ch].detach().cpu().numpy(), 
                                 aspect='equal', cmap='viridis')  # 改为正方形展示
        axes[1, ch].set_ylabel('Neural Net', fontsize=12, fontweight='bold')
        
        # SVD重构
        im3 = axes[2, ch].imshow(X_svd_spatial[:, :, ch].detach().cpu().numpy(), 
                                 aspect='equal', cmap='viridis')  # 改为正方形展示
        axes[2, ch].set_xlabel('Width', fontsize=10)
        axes[2, ch].set_ylabel('SVD', fontsize=12, fontweight='bold')
        
        # 设置刻度
        axes[0, ch].set_xticks([])
        axes[1, ch].set_xticks([])
        axes[2, ch].set_xticks([])
        
        # 只在最左边显示y轴标签
        if ch > 0:
            axes[0, ch].set_yticks([])
            axes[1, ch].set_yticks([])
            axes[2, ch].set_yticks([])
    
    # 添加总标题
    fig.suptitle(f'Spatial Feature Comparison (Image {image_idx}, Image: {image_size}x{image_size}, Patch: {patch_size}x{patch_size})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'image_{image_idx}_spatial_channel_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def visualize_differences_spatial(X: torch.Tensor, X_hat: torch.Tensor, X_svd: torch.Tensor, 
                                 out_dir: str, image_idx: int = 0, image_size: int = 448, patch_size: int = 16) -> None:
    """可视化重构误差，每个channel显示为16x16的空间布局"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 计算误差
    err_nn = (X - X_hat).detach()
    err_svd = (X - X_svd).detach()
    
    # 取前8个channel
    num_channels = min(8, X.shape[-1])
    
    # 重塑为空间布局
    err_nn_spatial = reshape_patches_to_spatial(err_nn, image_size, patch_size)
    err_svd_spatial = reshape_patches_to_spatial(err_svd, image_size, patch_size)
    
    # 创建2行8列的子图布局
    fig, axes = plt.subplots(2, num_channels, figsize=(20, 8))
    
    # 为每个channel创建误差可视化
    for ch in range(num_channels):
        # 神经网络重构误差
        im1 = axes[0, ch].imshow(np.abs(err_nn_spatial[:, :, ch].cpu().numpy()), 
                                 aspect='equal', cmap='magma')  # 改为正方形展示
        axes[0, ch].set_title(f'Ch{ch+1}', fontsize=12)
        if ch == 0:
            axes[0, ch].set_ylabel('Neural Net Error', fontsize=12, fontweight='bold')
        
        # SVD重构误差
        im2 = axes[1, ch].imshow(np.abs(err_svd_spatial[:, :, ch].cpu().numpy()), 
                                 aspect='equal', cmap='magma')  # 改为正方形展示
        if ch == 0:
            axes[1, ch].set_ylabel('SVD Error', fontsize=12, fontweight='bold')
        
        # 设置刻度
        axes[0, ch].set_xticks([])
        axes[1, ch].set_xticks([])
        
        # 只在最左边显示y轴标签
        if ch > 0:
            axes[0, ch].set_yticks([])
            axes[1, ch].set_yticks([])
    
    # 添加总标题
    fig.suptitle(f'Spatial Reconstruction Error (Image {image_idx}, Image: {image_size}x{image_size}, Patch: {patch_size}x{patch_size})', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'image_{image_idx}_spatial_error_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def compute_cosine_similarity(X_original, X_reconstructed):
    """计算重构矩阵与原始矩阵的余弦相似度"""
    # 将张量展平为2D矩阵 [batch_size, tokens*dim]
    X_flat = X_original.view(X_original.size(0), -1)
    X_recon_flat = X_reconstructed.view(X_reconstructed.size(0), -1)
    
    # 计算余弦相似度
    similarities = []
    for i in range(X_flat.size(0)):
        sim = np.dot(X_flat[i].cpu().numpy(), X_recon_flat[i].cpu().numpy()) / (
            np.linalg.norm(X_flat[i].cpu().numpy()) * np.linalg.norm(X_recon_flat[i].cpu().numpy()) + 1e-8
        )
        similarities.append(sim)
    
    return np.mean(similarities)


def visualize_single_image_comparison(dataset, model, rank, device, results_dir, image_idx=0):
    """可视化单张图像的重构比较 - 使用空间布局展示前8个通道"""
    print(f"🎨 可视化图像 {image_idx} 的重构比较...")
    
    # 获取图像数据
    X = dataset[image_idx][0].unsqueeze(0)  # [1, 256, 2048]
    print(f"📊 原始特征矩阵形状: {X.shape}")
    
    # 1. SVD重构
    print("📈 计算SVD重构...")
    X_svd_recon = svd_truncated_reconstruction(X.squeeze(0), rank).unsqueeze(0)
    
    # 2. 神经网络重构
    print("🧠 计算神经网络重构...")
    model.eval()
    with torch.no_grad():
        X_norm = layernorm_along_feature(X.to(device))
        X_hat, A, B = model(X_norm)
        X_nn_recon = X_hat
    
    print(f"✅ 重构完成:")
    print(f"  SVD重构形状: {X_svd_recon.shape}")
    print(f"  NN重构形状: {X_nn_recon.shape}")
    
    # 3. 计算余弦相似度
    svd_sim = compute_cosine_similarity(X, X_svd_recon)
    nn_sim = compute_cosine_similarity(X, X_nn_recon)
    
    print(f"📊 余弦相似度:")
    print(f"  SVD: {svd_sim:.4f}")
    print(f"  NN:  {nn_sim:.4f}")
    
    # 4. 生成空间布局的可视化
    print("🎨 生成空间布局可视化...")
    try:
        # 确保所有张量都在CPU上
        X_cpu = X.squeeze(0).cpu()
        X_nn_recon_cpu = X_nn_recon.squeeze(0).cpu()
        X_svd_recon_cpu = X_svd_recon.squeeze(0).cpu()
        
        # 使用空间布局展示前8个通道
        visualize_channels_spatial(X_cpu, X_nn_recon_cpu, X_svd_recon_cpu, 
                                 results_dir, image_idx, image_size=448, patch_size=16)
        visualize_differences_spatial(X_cpu, X_nn_recon_cpu, X_svd_recon_cpu, 
                                    results_dir, image_idx, image_size=448, patch_size=16)
        print(f"✅ 空间布局可视化生成完成")
    except Exception as e:
        print(f"⚠️  空间布局可视化生成失败: {e}")
        print("回退到原始热力图可视化...")
        
        # 回退到原始的热力图可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Image {image_idx} Reconstruction Comparison (Rank={rank})', fontsize=16)
        
        # 原始特征矩阵热力图
        im1 = axes[0, 0].imshow(X[0].cpu().numpy(), cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Original Features')
        axes[0, 0].set_xlabel('Feature Dimension')
        axes[0, 0].set_ylabel('Token Position')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # SVD重构热力图
        im2 = axes[0, 1].imshow(X_svd_recon[0].cpu().numpy(), cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'SVD Reconstruction\n(Similarity: {svd_sim:.4f})')
        axes[0, 1].set_xlabel('Feature Dimension')
        axes[0, 1].set_ylabel('Token Position')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 神经网络重构热力图
        im3 = axes[0, 2].imshow(X_nn_recon[0].cpu().numpy(), cmap='viridis', aspect='auto')
        axes[0, 2].set_title(f'Neural Network Reconstruction\n(Similarity: {nn_sim:.4f})')
        axes[0, 2].set_xlabel('Feature Dimension')
        axes[0, 2].set_ylabel('Token Position')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 重构误差热力图
        svd_error = torch.abs(X - X_svd_recon)[0].cpu().numpy()
        nn_error = torch.abs(X - X_nn_recon)[0].cpu().numpy()
        
        # 计算误差的统计信息用于显示
        svd_error_min, svd_error_max = svd_error.min(), svd_error.max()
        nn_error_min, nn_error_max = nn_error.min(), nn_error.max()
        
        im4 = axes[1, 0].imshow(svd_error, cmap='Reds', aspect='auto', vmin=0, vmax=svd_error_max)
        axes[1, 0].set_title(f'SVD Reconstruction Error\nMin: {svd_error_min:.4f}, Max: {svd_error_max:.4f}')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Token Position')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(nn_error, cmap='Reds', aspect='auto', vmin=0, vmax=nn_error_max)
        axes[1, 1].set_title(f'NN Reconstruction Error\nMin: {nn_error_min:.4f}, Max: {nn_error_max:.4f}')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Token Position')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # 相似度对比条形图
        methods = ['SVD', 'Neural Network']
        similarities = [svd_sim, nn_sim]
        colors = ['blue', 'red']
        
        bars = axes[1, 2].bar(methods, similarities, color=colors, alpha=0.7)
        axes[1, 2].set_title('Cosine Similarity Comparison')
        axes[1, 2].set_ylabel('Cosine Similarity')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # 在条形图上添加数值标签
        for bar, sim in zip(bars, similarities):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{sim:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存可视化结果
        vis_path = os.path.join(results_dir, f"image_{image_idx}_reconstruction_comparison.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"✅ 热力图可视化结果保存到: {vis_path}")
    
    # 保存数值结果
    results = {
        'image_idx': image_idx,
        'rank': rank,
        'similarities': {
            'svd': float(svd_sim),
            'neural_network': float(nn_sim)
        },
        'shapes': {
            'original': X.shape,
            'svd_reconstruction': X_svd_recon.shape,
            'nn_reconstruction': X_nn_recon.shape
        }
    }
    
    import json
    results_path = os.path.join(results_dir, f"image_{image_idx}_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 数值结果保存到: {results_path}")
    
    return results


def layernorm_along_feature(X: torch.Tensor) -> torch.Tensor:
    """沿着特征维度进行层归一化"""
    return torch.nn.functional.layer_norm(X, normalized_shape=(X.size(-1),))


def main():
    parser = argparse.ArgumentParser(description='Visualize SVD vs Neural Network reconstruction')
    parser.add_argument('--rank', type=int, default=128, help='Rank for decomposition')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--image_idx', type=int, default=0, help='Image index to visualize')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 检查模型文件
    model_path = os.path.join(args.results_dir, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 加载数据集
    print("📊 加载数据集...")
    dataset = RealQwenVLDataset(use_cache=True, cache_dir="./feature_cache_real")
    
    if args.image_idx >= len(dataset):
        print(f"❌ 图像索引 {args.image_idx} 超出范围 (0-{len(dataset)-1})")
        return
    
    # 加载训练好的模型
    print("🧠 加载训练好的模型...")
    model = LowRankProjSymmetric(input_dim=2048, num_tokens=256, rank=args.rank, normalize=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    # 可视化比较
    results = visualize_single_image_comparison(dataset, model, args.rank, device, args.results_dir, args.image_idx)
    
    print(f"\n🎉 可视化完成！")
    print(f"📊 结果:")
    print(f"  SVD相似度: {results['similarities']['svd']:.4f}")
    print(f"  NN相似度:  {results['similarities']['neural_network']:.4f}")


if __name__ == "__main__":
    main()
