#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from typing import Optional


class RankAllocatorWithUpdate:
    """
    简化版 AdaLoRA RankAllocator：
    - 基于梯度重要性为全局缩放向量 lora_E 生成掩码
    - 在训练中周期性更新并掩码，动态控制有效秩
    - 同步写回当前有效秩到 model.ranknum，并清理被掩参数的优化器动量
    使用方法：
      allocator = RankAllocatorWithUpdate(model, optimizer, init_warmup=1000, final_warmup=1000, mask_interval=20)
      ...
      loss.backward(); optimizer.step()
      allocator.update_and_mask(step)
      optimizer.zero_grad()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        init_warmup: int = 1000,
        final_warmup: int = 1000,
        mask_interval: int = 20,
        min_keep: int = 1,
        keep_ratio: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.init_warmup = int(init_warmup)
        self.final_warmup = int(final_warmup)
        self.mask_interval = int(mask_interval)
        self.min_keep = int(min_keep)
        self.keep_ratio = keep_ratio  # 若提供，则按比例保留

        # 仅针对存在 lora_E 的模型
        if not hasattr(model, "lora_E"):
            raise ValueError("RankAllocatorWithUpdate 需要模型包含 lora_E 参数")

        self.register_buffer_like_model()

    def register_buffer_like_model(self):
        E = self.model.lora_E.view(-1)
        self.r = E.numel()
        self.importance = torch.zeros_like(E, dtype=torch.float32)  # EMA 的重要性

    @torch.no_grad()
    def _collect_importance(self):
        # 简化：使用 lora_E 的梯度绝对值作为重要性
        if self.model.lora_E.grad is None:
            return
        g = self.model.lora_E.grad.view(-1).abs()
        if self.importance.device != g.device:
            self.importance = self.importance.to(g.device)
        # EMA 更新
        beta = 0.85
        self.importance.mul_(beta).add_(g, alpha=1.0 - beta)

    @torch.no_grad()
    def _mask_by_threshold(self, keep_k: int):
        keep_k = max(self.min_keep, min(int(keep_k), self.r))
        # 按重要性排序，保留 top-k
        scores = self.importance.clone()
        # 避免全 0 的情况退化
        if torch.all(scores == 0):
            scores = torch.arange(self.r, device=scores.device, dtype=scores.dtype)
        topk = torch.topk(scores, k=keep_k, largest=True).indices
        mask = torch.zeros(self.r, dtype=torch.bool, device=scores.device)
        mask[topk] = True

        # 应用到 lora_E：保留的通道为原值，其他置 0
        E = self.model.lora_E.view(-1)
        E[~mask] = 0.0

        # 写回 ranknum
        if hasattr(self.model, "ranknum") and isinstance(self.model.ranknum, torch.nn.Parameter):
            self.model.ranknum.data = torch.tensor(float(mask.sum().item()), device=E.device)

        # 清理被掩元素的优化器动量，避免残留影响
        state = self.optimizer.state
        for p in [self.model.lora_E]:
            if p in state:
                st = state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and isinstance(st[key], torch.Tensor):
                        st[key].view(-1)[~mask] = 0.0

        return mask

    @torch.no_grad()
    def update_and_mask(self, step: int):
        # 在 warmup 阶段积累重要性，不掩码
        if step < self.init_warmup:
            self._collect_importance()
            return None

        # mask 的时间点：间隔调用
        if (step - self.init_warmup) % max(1, self.mask_interval) != 0:
            self._collect_importance()
            return None

        # 线性从 init_warmup -> init_warmup+final_warmup 降低保留比例到目标
        total_warm = self.init_warmup + self.final_warmup
        if step >= total_warm:
            target_keep = self.min_keep if self.keep_ratio is None else int(self.r * float(self.keep_ratio))
        else:
            # 线性调度：由全部 -> 目标
            progress = (step - self.init_warmup) / max(1, self.final_warmup)
            full_keep = self.r
            tgt_keep = self.min_keep if self.keep_ratio is None else int(self.r * float(self.keep_ratio))
            target_keep = int(full_keep - progress * (full_keep - tgt_keep))

        self._collect_importance()
        mask = self._mask_by_threshold(target_keep)
        return mask


