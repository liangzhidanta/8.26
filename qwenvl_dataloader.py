#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实QwenVL3B模型的数据集加载器
按照test.ipynb中的方式获取图像特征
专门使用Qwen2.5-VL-3B模型
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ["MODELSCOPE_CACHE"] = "/data2/dzr/.cache" 
os.environ["TOKENS_PARALLELISM"] = "false"

# 导入必要的模块
try:
    from modelscope import AutoProcessor, AutoModel
    from qwen_vl_utils import process_vision_info
    print("Using ModelScope for model loading")
    MODEL_LOADER = "modelscope"
except ImportError as e:
    print(f"Warning: {e}")
    print("Trying alternative imports...")
    try:
        from transformers import AutoProcessor, AutoModel
        # 简单的fallback实现
        def process_vision_info(messages):
            image_inputs = []
            for msg in messages:
                for content in msg["content"]:
                    if content["type"] == "image":
                        image_inputs.append(content["image"])
            return image_inputs, []
        print("Using Transformers as fallback")
        MODEL_LOADER = "transformers"
    except ImportError as e2:
        print(f"Error: {e2}")
        raise


class RealQwenVLDataset(Dataset):
    """使用真实QwenVL3B模型的textVQA数据集加载器"""
    
    def __init__(self, 
                 dataset_dir: str = "/data2/dzr/textVQA_groundingtask_bbox",
                 image_size: int = 448,
                 use_cache: bool = True,
                 cache_dir: str = "./feature_cache_real"):
        """
        Args:
            dataset_dir: 数据集目录
            image_size: 图像尺寸
            use_cache: 是否使用特征缓存
            cache_dir: 特征缓存目录
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # 加载数据集
        self._load_dataset()
        
        # 初始化QwenVL模型和处理器
        self._init_model()
        
        # 预处理数据
        self._preprocess_data()
    
    def _load_dataset(self):
        """加载数据集文件"""
        print("Loading textVQA dataset...")
        
        # 训练数据文件
        train_files = [
            "data/train-00000-of-00003-d0560b4d0b4feac4.parquet",
            "data/train-00001-of-00003-0085a9b480196c92.parquet", 
            "data/train-00002-of-00003-ebe81cbdb50b653b.parquet"
        ]
        
        # 加载所有训练数据
        train_data = []
        for file_path in train_files:
            full_path = os.path.join(self.dataset_dir, file_path)
            if os.path.exists(full_path):
                print(f"Loading file: {file_path}")
                df = pd.read_parquet(full_path)
                train_data.append(df)
                print(f"  - Samples: {len(df)}")
        
        # 合并数据
        if train_data:
            self.df = pd.concat(train_data, ignore_index=True)
            print(f"✅ Dataset loaded successfully! Total samples: {len(self.df)}")
        else:
            raise FileNotFoundError("No training data files found")
    
    def _init_model(self):
        """初始化QwenVL模型和处理器"""
        print("Initializing QwenVL model...")
        
        try:
            # 专门使用Qwen2.5-VL-3B-Instruct模型
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            print(f"Loading model: {model_name}")
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            print("✅ Qwen2.5-VL-3B-Instruct model loaded successfully")
            
        except Exception as e:
            print(f"Error: Failed to load Qwen2.5-VL-3B-Instruct: {e}")
            print("This model is required for the project. Please check:")
            print("1. Internet connection")
            print("2. Model availability")
            print("3. Transformers version compatibility")
            raise RuntimeError(f"Failed to load required model: {model_name}")
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 获取设备信息
        self.device = next(self.model.parameters()).device
        print(f"Model device: {self.device}")
        
        # 检查模型是否有get_image_features方法
        if not hasattr(self.model, 'get_image_features'):
            print("Warning: Model doesn't have get_image_features method")
            print("Available methods:", [method for method in dir(self.model) if not method.startswith('_')])
            print("This may cause issues with feature extraction.")
    
    def _bytes_to_image(self, image_bytes) -> Optional[Image.Image]:
        """将字节数据转换为PIL图像"""
        try:
            if isinstance(image_bytes, dict) and 'bytes' in image_bytes:
                img_data = image_bytes['bytes']
            else:
                img_data = image_bytes
            
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            return image
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """将图像resize到指定尺寸"""
        return image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
    
    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        """使用QwenVL获取图像特征，按照test.ipynb的方式"""
        try:
            # 按照test.ipynb的方式准备消息和输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            
            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            try:
                # 使用process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception as e:
                print(f"process_vision_info failed: {e}, using fallback")
                # fallback处理
                image_inputs = [image]
                video_inputs = []
            
            # 使用处理器处理输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # 移动到模型设备
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 从处理后的输入中获取 pixel_values 和 image_grid_thw
            pixel_values = inputs['pixel_values'].to(self.model.dtype)
            image_grid_thw = inputs.get('image_grid_thw', torch.tensor([[1, 1, 1]]).to(self.device))
            if not isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = torch.tensor(image_grid_thw).to(self.device)
            image_grid_thw = image_grid_thw.to(torch.long)
            
            # 通过模型的 get_image_features 方法编码图像
            with torch.no_grad():
                try:
                    image_features_list = self.model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
                    image_features = torch.cat(image_features_list, dim=0)
                except AttributeError:
                    print("Model doesn't have get_image_features method, trying alternative...")
                    # 如果没有get_image_features方法，尝试其他方法
                    # 这里可以添加其他特征提取方法
                    raise RuntimeError("Model doesn't support get_image_features")
                
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    raise
            
            # 确保特征维度正确 [batch_size, 256, 2048]
            if len(image_features.shape) == 2:
                # 如果是 [256, 2048]，添加batch维度
                image_features = image_features.unsqueeze(0)
            
            return image_features
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            raise
    
    def _preprocess_data(self):
        """预处理数据，提取图像特征"""
        print("Preprocessing data and extracting features...")
        
        # 检查是否可以直接使用缓存的特征
        if self.use_cache and os.path.exists(self.cache_dir):
            # 检查缓存目录中的特征文件数量
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('features_') and f.endswith('.pt')]
            if len(cache_files) == len(self.df):
                # 检查第一个特征文件的形状
                try:
                    first_cache_path = os.path.join(self.cache_dir, 'features_0.pt')
                    if os.path.exists(first_cache_path):
                        first_features = torch.load(first_cache_path, map_location='cpu')
                        if first_features.shape == (1, 256, 2048):
                            print(f"✅ Found complete feature cache with {len(cache_files)} samples")
                            print(f"✅ First feature shape: {first_features.shape}, dtype: {first_features.dtype}")
                            print("🚀 Skipping image processing, loading all cached features...")
                            
                            # 直接加载所有缓存的特征
                            self.features = []
                            self.valid_indices = []
                            
                            for idx in range(len(self.df)):
                                cache_path = os.path.join(self.cache_dir, f"features_{idx}.pt")
                                try:
                                    features = torch.load(cache_path, map_location='cpu')
                                    self.features.append(features)
                                    self.valid_indices.append(idx)
                                except Exception as e:
                                    print(f"Failed to load cached features for sample {idx}: {e}")
                                    continue
                            
                            print(f"✅ Loaded {len(self.features)} cached features")
                            
                            # 转换为tensor
                            if self.features:
                                self.features = torch.cat(self.features, dim=0)  # [N, 256, 2048]
                                print(f"✅ Final features tensor shape: {self.features.shape}")
                                print(f"✅ Final features tensor dtype: {self.features.dtype}")
                                print(f"✅ Final features tensor device: {self.features.device}")
                            else:
                                raise RuntimeError("No valid features loaded from cache")
                            
                            return  # 直接返回，跳过后续的图像处理逻辑
                        else:
                            print(f"⚠️  First cached feature has wrong shape: {first_features.shape}, expected (1, 256, 2048)")
                    else:
                        print("⚠️  First cache file not found")
                except Exception as e:
                    print(f"⚠️  Error checking first cache file: {e}")
                print("🔄 Proceeding with normal image processing...")
        
        self.features = []
        self.valid_indices = []
        
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(self.df)}")
            
            # 获取图像
            image = self._bytes_to_image(row['image'])
            if image is None:
                continue
            
            # Resize图像
            image = self._resize_image(image)
            
            # 检查缓存
            cache_path = None
            if self.use_cache:
                cache_path = os.path.join(self.cache_dir, f"features_{idx}.pt")
                if os.path.exists(cache_path):
                    try:
                        features = torch.load(cache_path, map_location='cpu')  # 加载到CPU
                        # 调试信息：检查缓存的特征维度
                        if idx < 5:  # 只打印前5个样本的调试信息
                            print(f"🔍 DEBUG: Loaded cached features {idx}: shape={features.shape}, dtype={features.dtype}")
                        self.features.append(features)
                        self.valid_indices.append(idx)
                        continue
                    except Exception as e:
                        print(f"Failed to load cached features for sample {idx}: {e}")
                        pass
            
            # 提取特征
            try:
                features = self._get_image_features(image)
                
                # 调试信息：检查原始特征维度
                if idx < 5:
                    print(f"🔍 DEBUG: Raw features {idx}: shape={features.shape}, dtype={features.dtype}")
                
                # 确保特征维度正确 [1, 256, 2048]
                if len(features.shape) == 2:
                    features = features.unsqueeze(0)  # 添加batch维度
                elif len(features.shape) == 3 and features.shape[0] != 1:
                    # 如果batch维度不是1，取第一个
                    features = features[:1]
                
                # 将特征移到CPU并转换为float32以避免pin_memory问题
                features = features.cpu().to(torch.float32)
                
                # 调试信息：检查处理后的特征维度
                if idx < 5:
                    print(f"🔍 DEBUG: Processed features {idx}: shape={features.shape}, dtype={features.dtype}")
                
                # 保存到缓存
                if self.use_cache and cache_path:
                    torch.save(features, cache_path)
                
                self.features.append(features)
                self.valid_indices.append(idx)
                
            except Exception as e:
                print(f"Failed to extract features for sample {idx}: {e}")
                continue
        
        print(f"✅ Feature extraction completed! Valid samples: {len(self.features)}")
        
        # 转换为tensor
        if self.features:
            # 检查所有特征的维度一致性
            feature_shapes = [f.shape for f in self.features]
            unique_shapes = set(feature_shapes)
            print(f"🔍 DEBUG: Feature shapes found: {unique_shapes}")
            
            if len(unique_shapes) > 1:
                print(f"⚠️  WARNING: Inconsistent feature shapes detected!")
                print(f"   Expected: torch.Size([1, 256, 2048])")
                print(f"   Found: {unique_shapes}")
                
                # 统一所有特征到标准维度
                print("🔧 Fixing feature dimensions...")
                standardized_features = []
                for i, f in enumerate(self.features):
                    if f.shape != (1, 256, 2048):
                        if len(f.shape) == 2:
                            f = f.unsqueeze(0)
                        elif f.shape[0] > 1:
                            f = f[:1]
                        if f.shape[1] != 256 or f.shape[2] != 2048:
                            print(f"⚠️  Sample {i} has wrong dimensions: {f.shape}, skipping...")
                            continue
                        f = f.to(torch.float32)
                    standardized_features.append(f)
                
                self.features = standardized_features
                print(f"✅ Standardized features: {len(self.features)} samples")
            
            self.features = torch.cat(self.features, dim=0)  # [N, 256, 2048]
            print(f"✅ Final features tensor shape: {self.features.shape}")
            print(f"✅ Final features tensor dtype: {self.features.dtype}")
            print(f"✅ Final features tensor device: {self.features.device}")
        else:
            raise RuntimeError("No valid features extracted")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """返回特征矩阵"""
        return (self.features[idx],)  # 返回tuple以兼容现有代码
    
    def get_split_datasets(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """划分数据集为训练、验证、测试集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        
        train_set, val_set, test_set = random_split(
            self, [train_size, val_size, test_size]
        )
        
        return train_set, val_set, test_set


def create_data_loaders_real(dataset: RealQwenVLDataset, 
                            batch_size: int = 8,
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1,
                            num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 划分数据集
    train_set, val_set, test_set = dataset.get_split_datasets(train_ratio, val_ratio, test_ratio)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # 启用pin_memory以加速GPU训练
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True  # 启用pin_memory以加速GPU训练
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True  # 启用pin_memory以加速GPU训练
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试真实QwenVL数据集加载器
    print("Testing real QwenVL dataset loader...")
    
    try:
        dataset = RealQwenVLDataset(use_cache=True)
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # 测试特征提取
        sample_features = dataset[0]
        print(f"Sample features shape: {sample_features[0].shape}")
        
        # 测试数据划分
        train_set, val_set, test_set = dataset.get_split_datasets()
        print(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        # 测试数据加载器
        train_loader, val_loader, test_loader = create_data_loaders_real(dataset, batch_size=4)
        print(f"Data loaders created successfully")
        
        # 测试一个batch
        for batch in train_loader:
            X = batch[0]
            print(f"Batch shape: {X.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

