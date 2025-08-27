import torch


@torch.no_grad()
def svd_truncated_error(X: torch.Tensor, rank: int) -> float:
    """
    计算单样本 X (T, D) 的截断SVD最优秩-r相对F范数误差。
    返回相对误差 ||X - X_r||_F / ||X||_F
    """
    if X.dim() != 2:
        raise ValueError("X must be (T, D)")
    T, D = X.shape
    r = min(rank, T, D)
    # Using full_matrices=False for efficiency
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_trunc = S.clone()
    if r < S_trunc.numel():
        S_trunc[r:] = 0.0
    # Reconstruct X_r
    X_r = (U * S_trunc.unsqueeze(0)) @ Vh
    num = torch.linalg.norm(X - X_r)
    den = torch.linalg.norm(X) + 1e-12
    return (num / den).item()


@torch.no_grad()
def svd_truncated_reconstruction(X: torch.Tensor, rank: int) -> torch.Tensor:
    """
    计算单样本 X (T, D) 的截断SVD重构。
    返回重构后的矩阵 X_r
    """
    if X.dim() != 2:
        raise ValueError("X must be (T, D)")
    T, D = X.shape
    r = min(rank, T, D)
    # Using full_matrices=False for efficiency
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_trunc = S.clone()
    if r < S_trunc.numel():
        S_trunc[r:] = 0.0
    # Reconstruct X_r
    X_r = (U * S_trunc.unsqueeze(0)) @ Vh
    return X_r



