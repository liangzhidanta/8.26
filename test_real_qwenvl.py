#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实QwenVL数据集加载器（小样本版本）
"""

import os
import torch
import time

# 设置环境变量
os.environ["MODELSCOPE_CACHE"] = "/data2/dzr/.cache" 
os.environ["TOKENS_PARALLELISM"] = "false"

def test_real_qwenvl_small():
    """测试真实QwenVL数据集加载器（小样本）"""
    print("=" * 60)
    print("Testing Real QwenVL Dataset Loader (Small Sample)")
    print("=" * 60)
    
    try:
        # 导入数据集加载器
        from qwenvl_dataloader import RealQwenVLDataset
        
        print("✅ Real QwenVL dataset loader imported successfully")
        
        # 创建一个限制样本数的数据集类
        class LimitedRealQwenVLDataset(RealQwenVLDataset):
            def _preprocess_data(self):
                """只处理前10个样本进行快速测试"""
                print("Preprocessing limited data (first 10 samples)...")
                
                self.features = []
                self.valid_indices = []
                
                max_samples = 10  # 只处理前10个样本
                
                for idx, row in self.df.iterrows():
                    if idx >= max_samples:
                        break
                        
                    print(f"Processing sample {idx+1}/{max_samples}")
                    
                    # 获取图像
                    image = self._bytes_to_image(row['image'])
                    if image is None:
                        continue
                    
                    # Resize图像
                    image = self._resize_image(image)
                    
                    # 提取特征
                    try:
                        features = self._get_image_features(image)
                        self.features.append(features)
                        self.valid_indices.append(idx)
                        print(f"  ✅ Sample {idx+1} processed successfully")
                    except Exception as e:
                        print(f"  ❌ Sample {idx+1} failed: {e}")
                        continue
                
                print(f"✅ Limited preprocessing completed! Valid samples: {len(self.features)}")
                
                # 转换为tensor
                if self.features:
                    self.features = torch.cat(self.features, dim=0)  # [N, 256, 2048]
                    print(f"Features tensor shape: {self.features.shape}")
                else:
                    raise RuntimeError("No valid features extracted")
        
        # 测试数据集加载（只加载前10个样本）
        print("\n1. Testing dataset loading...")
        start_time = time.time()
        
        dataset = LimitedRealQwenVLDataset(
            dataset_dir="/data2/dzr/textVQA_groundingtask_bbox",
            image_size=448,
            use_cache=False,  # 禁用缓存进行测试
            cache_dir="./feature_cache_test"
        )
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f}s")
        print(f"   Total samples: {len(dataset)}")
        
        # 测试特征形状
        print("\n2. Testing feature extraction...")
        sample_features = dataset[0]
        print(f"✅ Sample features shape: {sample_features[0].shape}")
        print(f"   Expected: [256, 2048]")
        print(f"   Actual: {list(sample_features[0].shape)}")
        
        # 测试数据划分
        print("\n3. Testing dataset splitting...")
        train_set, val_set, test_set = dataset.get_split_datasets(
            train_ratio=0.8, 
            val_ratio=0.1, 
            test_ratio=0.1
        )
        print(f"✅ Dataset split successful:")
        print(f"   Train: {len(train_set)} samples")
        print(f"   Val: {len(val_set)} samples") 
        print(f"   Test: {len(test_set)} samples")
        
        # 测试数据加载器创建
        print("\n4. Testing data loader creation...")
        from qwenvl_dataloader import create_data_loaders_real
        train_loader, val_loader, test_loader = create_data_loaders_real(
            dataset, 
            batch_size=2,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=0  # 使用单进程避免多进程问题
        )
        print(f"✅ Data loaders created successfully")
        
        # 测试一个batch
        print("\n5. Testing batch loading...")
        for batch in train_loader:
            X = batch[0]
            print(f"✅ Batch loaded successfully:")
            print(f"   Batch shape: {X.shape}")
            print(f"   Expected: [2, 256, 2048]")
            print(f"   Actual: {list(X.shape)}")
            break
        
        # 测试设备兼容性
        print("\n6. Testing device compatibility...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 测试特征移动到设备
        X_device = X.to(device)
        print(f"✅ Features moved to {device} successfully")
        print(f"   Device tensor shape: {X_device.shape}")
        
        print("\n" + "="*60)
        print("✅ All tests passed! Real QwenVL dataset loader is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting real QwenVL dataset test (small sample)...")
    
    # 运行测试
    success = test_real_qwenvl_small()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("You can now proceed with full training using:")
        print("python train_real_data.py")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
    
    print("\nTest completed!")
