#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较SVD和神经网络低秩分解的结果
计算重构矩阵与原始矩阵的余弦相似度
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import argparse

from qwenvl_dataloader import RealQwenVLDataset, create_data_loaders_real
from model import LowRankProjSymmetric
from svd_baseline import svd_truncated_error, svd_truncated_reconstruction


def compute_cosine_similarity(X_original, X_reconstructed):
    """计算重构矩阵与原始矩阵的余弦相似度"""
    # 将张量展平为2D矩阵 [batch_size, tokens*dim]
    X_flat = X_original.view(X_original.size(0), -1)
    X_recon_flat = X_reconstructed.view(X_reconstructed.size(0), -1)
    
    # 计算余弦相似度
    similarities = []
    for i in range(X_flat.size(0)):
        sim = cosine_similarity(
            X_flat[i:i+1].cpu().numpy(), 
            X_recon_flat[i:i+1].cpu().numpy()
        )[0, 0]
        similarities.append(sim)
    
    return np.array(similarities)


def compare_methods(dataset, test_loader, rank, device, results_dir):
    """比较SVD和神经网络方法"""
    print("🔍 比较SVD和神经网络低秩分解结果...")
    
    # 获取测试数据
    test_data = []
    for batch in test_loader:
        X = batch[0]
        test_data.append(X)
        if len(test_data) >= 10:  # 只取前10个batch
            break
    
    X_test = torch.cat(test_data, dim=0)
    print(f"📊 测试数据形状: {X_test.shape}")
    
    # 1. SVD方法
    print("\n📈 计算SVD重构结果...")
    svd_reconstructions = []
    svd_similarities = []
    
    for i in range(X_test.size(0)):
        X_sample = X_test[i:i+1]  # [1, 256, 2048]
        
        # 使用SVD重构 - 需要去掉batch维度
        X_sample_2d = X_sample.squeeze(0)  # [256, 2048]
        X_svd_recon = svd_truncated_reconstruction(X_sample_2d, rank)
        X_svd_recon = X_svd_recon.unsqueeze(0)  # 恢复batch维度 [1, 256, 2048]
        svd_reconstructions.append(X_svd_recon)
        
        # 计算余弦相似度
        sim = compute_cosine_similarity(X_sample, X_svd_recon)
        svd_similarities.append(sim[0])
    
    svd_reconstructions = torch.cat(svd_reconstructions, dim=0)
    svd_similarities = np.array(svd_similarities)
    
    print(f"✅ SVD重构完成，平均余弦相似度: {svd_similarities.mean():.4f}")
    
    # 2. 神经网络方法
    print("\n🧠 计算神经网络重构结果...")
    
    # 加载训练好的模型
    model_path = os.path.join(results_dir, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    model = LowRankProjSymmetric(input_dim=2048, num_tokens=256, rank=rank, normalize=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    nn_reconstructions = []
    nn_similarities = []
    
    with torch.no_grad():
        for i in range(X_test.size(0)):
            X_sample = X_test[i:i+1].to(device)
            
            # 使用神经网络重构
            X_nn_recon, _, _ = model(X_sample)
            nn_reconstructions.append(X_nn_recon.cpu())
            
            # 计算余弦相似度
            sim = compute_cosine_similarity(X_sample, X_nn_recon.cpu())
            nn_similarities.append(sim[0])
    
    nn_reconstructions = torch.cat(nn_reconstructions, dim=0)
    nn_similarities = np.array(nn_similarities)
    
    print(f"✅ 神经网络重构完成，平均余弦相似度: {nn_similarities.mean():.4f}")
    
    # 3. 结果比较
    print("\n📊 结果比较:")
    print(f"  SVD平均余弦相似度: {svd_similarities.mean():.4f} ± {svd_similarities.std():.4f}")
    print(f"  NN平均余弦相似度:  {nn_similarities.mean():.4f} ± {nn_similarities.std():.4f}")
    
    # 4. 可视化比较
    print("\n🎨 生成可视化比较图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'SVD vs Neural Network Comparison (Rank={rank})', fontsize=16)
    
    # 余弦相似度分布
    axes[0, 0].hist(svd_similarities, alpha=0.7, label='SVD', bins=20, color='blue')
    axes[0, 0].hist(nn_similarities, alpha=0.7, label='Neural Network', bins=20, color='red')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Cosine Similarity Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 余弦相似度对比
    x_pos = np.arange(len(svd_similarities))
    axes[0, 1].bar(x_pos - 0.2, svd_similarities, 0.4, label='SVD', alpha=0.7, color='blue')
    axes[0, 1].bar(x_pos + 0.2, nn_similarities, 0.4, label='Neural Network', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Cosine Similarity by Sample')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 重构误差对比
    svd_errors = torch.mean((X_test - svd_reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
    nn_errors = torch.mean((X_test - nn_reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
    
    axes[1, 0].plot(svd_errors, label='SVD', marker='o', color='blue')
    axes[1, 0].plot(nn_errors, label='Neural Network', marker='s', color='red')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Reconstruction Error by Sample')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差分布
    axes[1, 1].hist(svd_errors, alpha=0.7, label='SVD', bins=20, color='blue')
    axes[1, 1].hist(nn_errors, alpha=0.7, label='Neural Network', bins=20, color='red')
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reconstruction Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存比较图
    comparison_path = os.path.join(results_dir, "svd_nn_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✅ 比较图保存到: {comparison_path}")
    
    # 保存数值结果
    results = {
        'svd_similarities': svd_similarities.tolist(),
        'nn_similarities': nn_similarities.tolist(),
        'svd_errors': svd_errors.tolist(),
        'nn_errors': nn_errors.tolist(),
        'summary': {
            'svd_mean_similarity': float(svd_similarities.mean()),
            'svd_std_similarity': float(svd_similarities.std()),
            'nn_mean_similarity': float(nn_similarities.mean()),
            'nn_std_similarity': float(nn_similarities.std()),
            'svd_mean_error': float(svd_errors.mean()),
            'svd_std_error': float(svd_errors.std()),
            'nn_mean_error': float(nn_errors.mean()),
            'nn_std_error': float(nn_errors.std()),
        }
    }
    
    import json
    results_path = os.path.join(results_dir, "svd_nn_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 数值结果保存到: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare SVD and Neural Network results')
    parser.add_argument('--rank', type=int, default=128, help='Rank for decomposition')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")
    
    # 加载数据集
    print("📊 加载数据集...")
    dataset = RealQwenVLDataset(use_cache=True, cache_dir="./feature_cache_real")
    
    # 创建测试数据加载器
    _, _, test_loader = create_data_loaders_real(
        dataset, 
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=4
    )
    
    # 比较方法
    results = compare_methods(dataset, test_loader, args.rank, device, args.results_dir)
    
    print("\n🎉 比较完成！")


if __name__ == "__main__":
    main()
