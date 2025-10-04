import torch 
import math 
import triton 
import triton.language as tl 

@triton.jit
def flash_atten_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    Q_stride_batch, Q_stride_seq, Q_stride_dim, # 每个维度的步长
    K_stride_batch, K_stride_seq, K_stride_dim,
    V_stride_batch, V_stride_seq, V_stride_dim,
    O_stride_batch, O_stride_seq, O_stride_dim,
    L_stride_batch, L_stride_seq,
    q_len, k_len, h_dim, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Tile & batch index
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Compute base offsets for Q, K, V, O, and L
    q_offset = batch_index * Q_stride_batch + query_tile_index * Q_TILE_SIZE * Q_stride_seq
    k_offset = batch_index * K_stride_batch
    v_offset = batch_index * V_stride_batch
    o_offset = batch_index * O_stride_batch + query_tile_index * Q_TILE_SIZE * O_stride_seq
    l_offset = batch_index * L_stride_batch + query_tile_index * Q_TILE_SIZE * L_stride_seq
    
    # Create block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + q_offset, 
        shape=(q_len, D), 
        strides=(Q_stride_seq, Q_stride_dim), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0))
    K_base_ptr = K_ptr + k_offset
    V_base_ptr = V_ptr + v_offset
    O_block_ptr = tl.make_block_ptr(
        O_ptr + o_offset, 
        shape=(q_len, D), 
        strides=(O_stride_seq, O_stride_dim), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0))
    L_block_ptr = tl.make_block_ptr(
        L_ptr + l_offset, 
        shape=(q_len,), 
        strides=(L_stride_seq,), 
        offsets=(0,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,))

    # Initialize accumulators
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    # load Q_i
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # Loop over key tiles
    T_k = tl.cdiv(k_len, K_TILE_SIZE)
    for j in range(T_k):
        # Load K_j
        k_tile_start = j * K_TILE_SIZE
        k_tile_end = min((j + 1) * K_TILE_SIZE, k_len)
        K_block_ptr = tl.make_block_ptr(K_base_ptr, (D, k_len), (K_stride_dim, K_stride_seq), (0, k_tile_start), (D, K_TILE_SIZE), (0, 1))
        V_block_ptr = tl.make_block_ptr(V_base_ptr, (k_len, D), (V_stride_seq, V_stride_dim), (k_tile_start, 0), (K_TILE_SIZE, D), (1, 0))

        K_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scale

        # Causal masking
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        # Numerically stable softmax update
        m_i_new = tl.maximum(m_i, tl.max(S_ij, 1))
        exp_m_diff = tl.exp(m_i - m_i_new)
        P_ij_tilde = tl.exp(S_ij - m_i_new[:, None])
        l_i = l_i * exp_m_diff + tl.sum(P_ij_tilde, 1)
        o_i = o_i * exp_m_diff[:, None] + tl.dot(P_ij_tilde.to(V_j.dtype), V_j)
        m_i = m_i_new
    
    # Normalize output
    O_i_normalized = o_i / l_i[:, None]
    tl.store(O_block_ptr, O_i_normalized.to(O_ptr.type.element_ty), boundary_check=(0, 1))

    # Store log-sum-exp result
    L_i = m_i + tl.log(l_i)
    tl.store(L_block_ptr, L_i, boundary_check=(0,))
    
# -------------------------------
# Backward Pass (PyTorch)
# -------------------------------
@torch.compile(fullgraph=True)
def _flash_attention_backward_compiled(
    Q, K, V, O, L, grad_O, is_causal,
    batch_size, N_q, N_k, d_head,
    B_q, B_k
):
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    scale = d_head ** -0.5
    D = torch.sum(grad_O * O, dim=-1)

    T_q = math.ceil(N_q / B_q)
    T_k = math.ceil(N_k / B_k)

    for b in range(batch_size):
        for i in range(T_q):
            q_start, q_end = i * B_q, min((i + 1) * B_q, N_q)
            Q_i, dO_i, L_i, D_i = Q[b, q_start:q_end], grad_O[b, q_start:q_end], L[b, q_start:q_end], D[b, q_start:q_end]
            for j in range(T_k):
                k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)
                K_j, V_j = K[b, k_start:k_end], V[b, k_start:k_end]
                S_ij = (Q_i @ K_j.T) * scale

                if is_causal:
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(k_start, k_end, device=K.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    S_ij = torch.where(causal_mask, S_ij, -float('inf'))

                P_ij = torch.exp(S_ij - L_i.unsqueeze(1))
                dV[b, k_start:k_end] += P_ij.T @ dO_i
                dP_ij = dO_i @ V_j.T
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(1))
                dQ[b, q_start:q_end] += (dS_ij * scale) @ K_j
                dK[b, k_start:k_end] += (dS_ij * scale).T @ Q_i

    return dQ, dK, dV

class FlashAttentionTritonImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        batch_size, q_len, head_dim = Q.shape
        _, k_len, _ = K.shape
        scale = head_dim ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, q_len), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        T_q = math.ceil(q_len / Q_TILE_SIZE)
        grid = (T_q, batch_size)

        flash_atten_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            q_len, k_len, head_dim, scale,
            D=head_dim, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size, q_len, head_dim = Q.shape
        _, N_k, _ = K.shape

        B_q, B_k = 64, 64

        dQ, dK, dV = _flash_attention_backward_compiled(
            Q, K, V, O, L, grad_output, is_causal,
            batch_size, q_len, N_k, head_dim,
            B_q, B_k
        )
        return dQ, dK, dV, None