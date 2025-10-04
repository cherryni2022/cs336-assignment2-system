import torch
from einops import rearrange, einsum
import math

"""
torch.autograd.Function:使自定义操作能够无缝融入PyTorch的自动梯度计算图中
"""
class FlashAttentionPytorchImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        # 保证q, k, v的d_model相同
        assert q.shape[2] == k.shape[2] == v.shape[2], "q, k, v must have the same d_model"
        batch, q_len, d_model = q.shape
        _, k_len, _ = k.shape
        device = q.device

        # 1. 基于tile size 分块
        q_tile = 64
        k_tile = 64
        num_q_tiles = math.ceil(q_len / q_tile)
        num_k_tiles = math.ceil(k_len / k_tile)

        # 2. 以tite为单位迭代计算每个tite的 softmax(Q@K)@V
        # output: 最终计算结果, L: log-sum-exp values
        O = torch.empty((batch, q_len, d_model), device=device, dtype=torch.float32) # Output
        L = torch.empty((batch, q_len), device=device, dtype=torch.float32) # Log-sum-exp values

        for one_data in range(batch):
            Q_b = q[one_data]
            K_b = k[one_data]
            V_b = v[one_data]
            # Flash Attention Loop 
            for i in range(num_q_tiles):
                q_tile_start, q_tile_end = i * q_tile, min((i + 1) * q_tile, q_len)
                curr_q_size = q_tile_end - q_tile_start
                # Load Q_i 
                Q_i = Q_b[q_tile_start:q_tile_end, :]
                # Initialize O_i (attention的整体计算结果)
                # L_i(scores), M_i (计算过程中 max)
                O_i = torch.zeros((curr_q_size, d_model), device=device, dtype=torch.float32)
                # L_i 迭代过程中所有 exp(score_i - max)的sum, softmax的分母
                L_i = torch.zeros((curr_q_size,), device=device, dtype=torch.float32)
                M_i = torch.full((curr_q_size,), -float('inf'),device=device, dtype=torch.float32)

                for j in range(num_k_tiles):
                    k_tile_start, k_tile_end = j * k_tile, min((j + 1) * k_tile, k_len)
                    K_j = K_b[k_tile_start:k_tile_end, :]
                    V_j = V_b[k_tile_start:k_tile_end, :]

                    # Compute tile of pre-softmax attention scores
                    scale = d_model**-0.5
                    S_ij = einsum(Q_i, K_j, "q_tile_len d_model, k_tile_len d_model -> q_tile_len k_tile_len") * scale
                    
                    if causal:
                        q_indices = torch.arange(q_tile_start, q_tile_end, device=device)
                        k_indices = torch.arange(k_tile_start, k_tile_end, device=device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                    
                    # Compute new m_ij 
                    m_i_new = torch.maximum(M_i, S_ij.max(dim=1).values)

                    # Compute new p_ij, 即softmax的分子 exp(score_i - max)
                    P_ij = torch.exp(S_ij - m_i_new[:, None])

                    # compute the diff between curr tile max and prev tile max
                    delta_exp_m = torch.exp(M_i - m_i_new)

                    # Compute L_i
                    L_i = delta_exp_m * L_i + P_ij.sum(dim=1)

                    # Compute O_i
                    O_i = torch.diag(delta_exp_m) @ O_i + (P_ij @ V_j)
                    
                    # Update M_i
                    M_i = m_i_new
                    
                # 3. 计算最终结果
                O_i = torch.diag(1.0 / L_i) @ O_i
                O[one_data, q_tile_start:q_tile_end, :] = O_i.to(q.dtype)
                L[one_data, q_tile_start:q_tile_end] = M_i + torch.log(L_i)
        
        ctx.save_for_backward(q, k, v, O, L)
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.causal

        dQ, dK, dV = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        batch_size, N_q, d_head = q.shape
        _, N_k, _ = k.shape
        scale = d_head ** -0.5
        
        D = torch.sum(grad_output * O, dim=-1)
        B_q, B_k = 64, 64
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k)
        
        
        for b in range(batch_size):
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
                Q_i, dO_i, L_i, D_i = q[b, q_start:q_end, :], grad_output[b, q_start:q_end, :], L[b, q_start:q_end], D[b, q_start:q_end]
                for j in range(T_k):
                    k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                    K_j, V_j = k[b, k_start:k_end, :], v[b, k_start:k_end, :]
                    S_ij = (Q_i @ K_j.T) * scale
                    
                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device=q.device)
                        k_indices = torch.arange(k_start, k_end, device=k.device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                        
                    P_ij = torch.exp(S_ij - L_i.unsqueeze(1))
                    
                    dV[b, k_start:k_end, :] += P_ij.T @ dO_i
                    dP_ij = dO_i @ V_j.T
                    dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
                    dQ[b, q_start:q_end, :] += (dS_ij * scale) @ K_j
                    dK[b, k_start:k_end, :] += (dS_ij * scale).T @ Q_i
                    
        return dQ, dK, dV, None
