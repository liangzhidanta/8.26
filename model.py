import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Residual MLP block with LN + 2-layer MLP"""
    def __init__(self, dim, hidden, out=None):
        super().__init__()
        out = out or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Linear(dim, out) if dim != out else nn.Identity()

    def forward(self, x):
        h = self.act(self.fc1(self.ln(x)))
        h = self.fc2(h)
        return h + self.shortcut(x)


class LowRankProjSymmetric(nn.Module):
    """
    改进版：A 从行方向学习，B 从列方向学习 (对称)。
    输入支持 (T, D) 或 (B, T, D)。
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_tokens: int = 256,
        rank: int = 128,
        row_hidden_dim: int = 512,
        col_hidden_dim: int = 512,
        normalize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.eps = eps

        # row_net: each token -> r-dim
        self.row_net = nn.Sequential(
            ResidualMLP(input_dim, row_hidden_dim, row_hidden_dim),
            nn.ReLU(inplace=True),
            ResidualMLP(row_hidden_dim, row_hidden_dim, rank),
        )

        # col_net: each feature dim across tokens -> r-dim
        self.col_net = nn.Sequential(
            ResidualMLP(num_tokens, col_hidden_dim, col_hidden_dim),
            nn.ReLU(inplace=True),
            ResidualMLP(col_hidden_dim, col_hidden_dim, rank),
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
