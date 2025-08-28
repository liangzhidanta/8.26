import torch
import torch.nn as nn
import math


class ResidualMLP(nn.Module):
    """修复维度问题的残差MLP块"""
    def __init__(self, dim, hidden, out=None, num_sub_layers=2, dropout=0.1):
        super().__init__()
        out = out or dim
        self.num_sub_layers = num_sub_layers
        
        # 输入规范化
        self.ln_in = nn.LayerNorm(dim)
        
        # 主网络 - 确保维度正确传递
        layers = []
        prev_dim = dim
        for i in range(num_sub_layers):
            # 中间层保持hidden维度，最后一层映射到目标维度
            if i == num_sub_layers - 1:
                layers.append(nn.Linear(prev_dim, out))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden  # 更新当前维度
        
        # 输出规范化
        layers.append(nn.LayerNorm(out))
        
        self.main_net = nn.Sequential(*layers)
        
        # shortcut连接
        self.shortcut = nn.Linear(dim, out) if dim != out else nn.Identity()
        
        # 输出激活
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.ln_in(x)
        h = self.main_net(h)
        h = h + self.shortcut(x)
        return self.act(h)


class LowRankProjSymmetric(nn.Module):
    """修复维度匹配问题的低秩投影网络"""

    def __init__(
        self,
        input_dim: int = 2048,
        num_tokens: int = 256,
        rank: int = 128,
        row_hidden_dim: int = 1024,
        col_hidden_dim: int = 1024,
        normalize: bool = True,
        eps: float = 1e-6,
        num_mlp_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.eps = eps

        # 构建行网络 - 确保维度正确传递
        row_layers = []
        current_dim = input_dim
        for i in range(num_mlp_blocks):
            # 最后一个块输出到rank维度，其他保持hidden维度
            output_dim = rank if i == num_mlp_blocks - 1 else row_hidden_dim
            row_layers.append(
                ResidualMLP(
                    dim=current_dim,
                    hidden=row_hidden_dim,
                    out=output_dim,
                    num_sub_layers=2,
                    dropout=dropout
                )
            )
            current_dim = output_dim  # 关键：更新当前维度为输出维度
        
        self.row_net = nn.Sequential(*row_layers)

        # 构建列网络 - 确保维度正确传递
        col_layers = []
        current_dim = num_tokens
        for i in range(num_mlp_blocks):
            # 最后一个块输出到rank维度，其他保持hidden维度
            output_dim = rank if i == num_mlp_blocks - 1 else col_hidden_dim
            col_layers.append(
                ResidualMLP(
                    dim=current_dim,
                    hidden=col_hidden_dim,
                    out=output_dim,
                    num_sub_layers=2,
                    dropout=dropout
                )
            )
            current_dim = output_dim  # 关键：更新当前维度为输出维度
        
        self.col_net = nn.Sequential(*col_layers)

        # 精炼层 - 确保输入输出维度一致
        self.refine_layer = nn.Sequential(
            ResidualMLP(input_dim, input_dim//2, input_dim, num_sub_layers=1),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, X):
        """保持原有接口"""
        squeezed = False
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, T, D)
            squeezed = True

        Bsz, T, D = X.shape
        r = self.rank

        # 行网络处理
        A = self.row_net(X)  # (B, T, r)

        # 列网络处理
        X_T = X.transpose(1, 2)  # (B, D, T)
        B_col = self.col_net(X_T)  # (B, D, r)
        B = B_col.transpose(1, 2)  # (B, r, D)

        if self.normalize:
            B = B / (B.norm(dim=-1, keepdim=True) + self.eps)
            A = A * (A.norm(dim=-1, keepdim=True) + self.eps)

        # 重构
        X_hat = torch.einsum("btr,brd->btd", A, B)
        
        # 精炼输出
        X_hat = self.refine_layer(X_hat)

        if squeezed:
            X_hat = X_hat.squeeze(0)
            A = A.squeeze(0)
            B = B.squeeze(0)

        return X_hat, A, B
    

class SVDLinear(nn.Module):
    """
    简化版 SVD/LoRA 线性层：y = x W^T + (alpha/r) * x (B^T A^T)
    - base_weight: [out, in]
    - lora_A: [in, r], lora_B: [out, r]
    - 可选 lora_dropout
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / max(r, 1)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in)
        base = torch.matmul(x, self.weight.t())
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            x_drop = self.dropout(x)
            # x_drop [*, in] @ A [in, r] -> [*, r]
            low = torch.matmul(x_drop, self.lora_A)
            # [*, r] @ B^T [r, out] -> [*, out]
            low = torch.matmul(low, self.lora_B.t()) * self.scaling
            return base + low
        return base


class LowRankProjMLP(nn.Module):
    """
    改进版：A 从行方向学习，B 从列方向学习 (对称)。
    输入支持 (T, D) 或 (B, T, D)。
    使用4层MLP：行网络 2048-1024-256-128，列网络 256-512-256-128
    添加ReLU激活函数、LayerNorm和Dropout提升性能
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_tokens: int = 256,
        rank: int = 128,
        normalize: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.eps = eps
        self.dropout = dropout

        # 构建行网络：2048 -> 1024 -> 256 -> 128
        self.row_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, rank)
        )
        
        # 构建列网络：256 -> 512 -> 256 -> 128
        self.col_net = nn.Sequential(
            nn.Linear(num_tokens, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, rank)
        )

    def forward(self, X):
        """
        X: (T, D) or (B, T, D)
        Returns:
          X_hat: same shape as X
          A: (B, T, r)
          B: (B, r, D)
        """
        squeezed = False
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, T, D)
            squeezed = True

        Bsz, T, D = X.shape
        r = self.rank

        # ----- A -----
        A = self.row_net(X)  # (B, T, r)

        # ----- B -----
        # Transpose: (B, T, D) -> (B, D, T)
        X_T = X.transpose(1, 2)
        B_col = self.col_net(X_T)  # (B, D, r)
        B = B_col.transpose(1, 2)  # (B, r, D)

        if self.normalize:
            # normalize rows of B
            B = B / (B.norm(dim=-1, keepdim=True) + self.eps)

        # reconstruct
        X_hat = torch.einsum("btr,brd->btd", A, B)

        if squeezed:
            X_hat = X_hat.squeeze(0)
            A = A.squeeze(0)
            B = B.squeeze(0)

        return X_hat, A, B


def get_model_by_name(
    arch_name: str,
    *,
    input_dim: int,
    num_tokens: int,
    rank: int,
    normalize: bool = False,
    **kwargs
):
    """根据名称创建模型，支持 'ResMLP' 与 'MLP'。"""
    name = (arch_name or "").lower()
    if name in ("resmlp", "lowrankprojsymmetric", "symmetric"):
        return LowRankProjSymmetric(
            input_dim=input_dim,
            num_tokens=num_tokens,
            rank=rank,
            normalize=normalize,
            **{k: v for k, v in kwargs.items() if k in {
                "row_hidden_dim", "col_hidden_dim", "num_mlp_blocks", "dropout", "eps"
            }}
        )
    if name in ("mlp", "lowrankprojmlp"):
        return LowRankProjMLP(
            input_dim=input_dim,
            num_tokens=num_tokens,
            rank=rank,
            normalize=normalize,
            **{k: v for k, v in kwargs.items() if k in {"eps", "dropout"}}
        )
    if name in ("adalora", "adaptive_adalora", "adaptive"):
        return AdaptiveLowRankProjAdaLoRA(
            input_dim=input_dim,
            num_tokens=num_tokens,
            rank=rank,
            normalize=normalize,
            **{k: v for k, v in kwargs.items() if k in {"eps", "dropout", "lora_r", "lora_alpha", "lora_dropout"}}
        )
    # 默认
    return LowRankProjSymmetric(input_dim=input_dim, num_tokens=num_tokens, rank=rank, normalize=normalize)


class AdaptiveLowRankProjAdaLoRA(nn.Module):
    """
    AdaLoRA 风格的低秩自适应投影：
    - 使用 SVDLinear 构建行/列网络
    - 增加全局可训练缩放向量 lora_E ∈ R^{r×1}，控制有效秩
    - forward 中 A = A_raw * E，B = B_raw（可选也乘 E）
    - 提供 effective_rank_mask / current_effective_rank 便于外部 RankAllocator 集成
    """
    def __init__(
        self,
        input_dim: int = 2048,
        num_tokens: int = 256,
        rank: int = 128,
        normalize: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.eps = eps

        # 行网络: 2048 -> 1024 -> 256 -> r
        self.row_net = nn.Sequential(
            SVDLinear(input_dim, 1024, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SVDLinear(1024, 256, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SVDLinear(256, rank, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
        )

        # 列网络: 256 -> 512 -> 256 -> r  （作用在 X^T 的 token 维）
        self.col_net = nn.Sequential(
            SVDLinear(num_tokens, 512, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SVDLinear(512, 256, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            SVDLinear(256, rank, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
        )

        # 全局可掩码奇异值向量（默认全1）
        self.lora_E = nn.Parameter(torch.ones(rank, 1))
        self.ranknum = nn.Parameter(torch.tensor(float(rank)), requires_grad=False)

        self.refine = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, X: torch.Tensor):
        squeezed = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            squeezed = True

        # A_raw: (B, T, r)
        A_raw = self.row_net(X)
        # B_raw: (B, r, D)
        X_T = X.transpose(1, 2)
        B_raw = self.col_net(X_T).transpose(1, 2)

        # 逐通道缩放 E
        E = self.lora_E.view(1, 1, -1)  # (1,1,r)
        A = A_raw * E
        B = B_raw  # 如需也缩放：B = B_raw * self.lora_E.view(1, -1, 1)

        if self.normalize:
            B = B / (B.norm(dim=-1, keepdim=True) + self.eps)

        X_hat = torch.einsum("btr,brd->btd", A, B)
        X_hat = self.refine(X_hat)

        if squeezed:
            X_hat = X_hat.squeeze(0)
            A = A.squeeze(0)
            B = B.squeeze(0)

        return X_hat, A, B

    def effective_rank_mask(self) -> torch.Tensor:
        return (self.lora_E.view(-1) != 0.0)

    def current_effective_rank(self) -> int:
        return int((self.lora_E.view(-1) != 0.0).sum().item())