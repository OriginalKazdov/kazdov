"""Kazdov-LM: autoregressive language model with BCN_hybrid attention.

This is the autoregressive transition for Kazdov. Math reasoning requires
generating multi-step solutions token-by-token, not classifying.

Architecture per block:
  - Causal BCN-attention (bilinear pairwise, masked to j <= i)
  - + Causal MHA (standard transformer attention)
  - + FFN (GELU)
  - + LayerNorm pre-norm + residual

The forward returns logits (B, T, V) suitable for next-token prediction loss.
The generate() method does standard autoregressive sampling.

Designed to be HuggingFace-compatible-ish (no full PreTrainedModel inheritance
for now — keep simple, can add later if needed).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class BilinearComposition(nn.Module):
    """Same primitive: out = U @ ((V^T x) ⊙ (W^T y)) + bias."""
    def __init__(self, d_in, d_out, rank, init_scale=0.02):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_out, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(d_in, rank) * init_scale)
        self.W = nn.Parameter(torch.randn(d_in, rank) * init_scale)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, y):
        return ((x @ self.V) * (y @ self.W)) @ self.U.T + self.bias


class CausalBCNAttention(nn.Module):
    """BCN attention with causal masking (autoregressive).

    For each token i, compose with all previous tokens j <= i:
        out_i = (1/n_i) * sum_{j<=i} BCN(Q_i, K_j)
    where Q = W_Q @ x, K = W_K @ x.
    """
    def __init__(self, d_model, rank, dropout=0.0):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.bcn = BilinearComposition(d_model, d_model, rank)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        x: (B, T, D)
        attention_mask: (B, T) bool, True = real token (not PAD). Optional.
        """
        B, T, D = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        Q_i = Q.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)  # (B*T*T, D)
        K_j = K.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)
        out = self.bcn(Q_i, K_j).reshape(B, T, T, D)  # (B, T, T, D)

        # Causal mask: j <= i
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)
        causal_mask = causal.unsqueeze(0).unsqueeze(-1).float()  # (1, T, T, 1)
        out = out * causal_mask

        # Padding mask (j-side only)
        if attention_mask is not None:
            pad_j = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            out = out * pad_j
            # Count valid j per i
            n_valid = (causal.unsqueeze(0).float() * attention_mask.unsqueeze(1).float())  # (B, T, T)
        else:
            n_valid = causal.unsqueeze(0).float().expand(B, T, T)
        n_valid_per_i = n_valid.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, T, 1)
        out = out.sum(dim=2) / n_valid_per_i  # (B, T, D)
        return self.dropout(self.W_O(out))


class CausalMHA(nn.Module):
    """Standard multi-head attention with causal mask, returns same shape as input."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, attention_mask=None):
        B, T, _ = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        kp_mask = ~attention_mask if attention_mask is not None else None
        out, _ = self.mha(x, x, x, attn_mask=causal_mask, key_padding_mask=kp_mask, need_weights=False)
        return out


class HybridCausalAttention(nn.Module):
    """BCN-causal + MHA-causal in parallel."""
    def __init__(self, d_model, n_heads, rank, dropout=0.0):
        super().__init__()
        self.bcn = CausalBCNAttention(d_model, rank, dropout=dropout)
        self.mha = CausalMHA(d_model, n_heads, dropout=dropout)

    def forward(self, x, attention_mask=None):
        return self.bcn(x, attention_mask) + self.mha(x, attention_mask)


class CausalTrilinearBCNAttention(nn.Module):
    """Bilinear + Trilinear composition with causal masking.

    Bilinear: out_i_b = sum_{j<=i} BCN(Q_i, K_j) [same as before]
    Trilinear: out_i_t = sum_{j<=i} TriBCN(Q_i, K_j, Z_i)
        where Z_i = causal cumulative mean of x at position i (only past tokens).

    Combined: out = bilinear + alpha * trilinear, alpha learnable.

    Validated externally by TrilinearCIM (arXiv:2604.07628): trilinear > bilinear
    on 7/9 GLUE tasks. We extend to causal autoregressive setting.
    """
    def __init__(self, d_model, rank, init_alpha=0.5, dropout=0.0):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        # Bilinear branch
        self.U_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias_b = nn.Parameter(torch.zeros(d_model))
        # Trilinear branch
        self.U_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.X_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias_t = nn.Parameter(torch.zeros(d_model))
        # Learnable alpha (mixing weight for trilinear)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)

        # Causal cumulative mean for trilinear context (per-position, only past tokens)
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            masked_x = x * mask_f
            cum_x = masked_x.cumsum(dim=1)  # (B, T, D)
            cum_count = mask_f.cumsum(dim=1).clamp(min=1.0)  # (B, T, 1)
            x_global_causal = cum_x / cum_count  # (B, T, D)
        else:
            cum_x = x.cumsum(dim=1)
            counts = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1, T, 1)
            x_global_causal = cum_x / counts

        # Pairwise tensors
        Q_i = Q.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)  # (B*T*T, D), per-i query
        K_j = K.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)  # (B*T*T, D), per-j key
        # Trilinear context: per-i causal global, broadcast over j
        Z_i = x_global_causal.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)

        # Bilinear branch
        bil = ((Q_i @ self.V_b) * (K_j @ self.W_b)) @ self.U_b.T + self.bias_b
        # Trilinear branch
        tri = ((Q_i @ self.V_t) * (K_j @ self.W_t) * (Z_i @ self.X_t)) @ self.U_t.T + self.bias_t

        out = (bil + self.alpha * tri).reshape(B, T, T, D)

        # Causal mask: j <= i
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)
        causal_f = causal.unsqueeze(0).unsqueeze(-1).float()  # (1, T, T, 1)
        out = out * causal_f

        # Padding mask (j-side)
        if attention_mask is not None:
            pad_j = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            out = out * pad_j
            n_valid = (causal.unsqueeze(0).float() * attention_mask.unsqueeze(1).float())  # (B, T, T)
        else:
            n_valid = causal.unsqueeze(0).float().expand(B, T, T)
        n_valid_per_i = n_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, T, 1)
        out = out.sum(dim=2) / n_valid_per_i  # (B, T, D)

        return self.dropout(self.W_O(out))


class HybridCausalTrilinearAttention(nn.Module):
    """Causal BCN+Trilinear + Causal MHA in parallel."""
    def __init__(self, d_model, n_heads, rank, init_alpha=0.5, dropout=0.0):
        super().__init__()
        self.bcn_tri = CausalTrilinearBCNAttention(d_model, rank, init_alpha=init_alpha, dropout=dropout)
        self.mha = CausalMHA(d_model, n_heads, dropout=dropout)

    def forward(self, x, attention_mask=None):
        return self.bcn_tri(x, attention_mask) + self.mha(x, attention_mask)


class MixtureBilinear(nn.Module):
    """Mixture of K bilinear experts with learned router (MoBE primitive).

    Each expert is a low-rank bilinear: out_k = U_k ((V_k^T x) ⊙ (W_k^T y)) + b_k.
    Router: small MLP over [x, y] outputs softmax weights over K experts.

    Vectorized implementation: parameters stacked as (K, d, r) tensors so all K
    experts compute in a single batched matmul — no Python loop. Roughly K×
    faster than the naive ModuleList form at LM scale.

    Balanced routing (DeepSeek-V3 style): dynamic per-expert bias added to
    router logits, updated externally by update_bias() to push usage toward
    uniform. No aux loss needed.
    """
    def __init__(self, d_in, d_out, rank, n_experts=4, init_scale=0.02,
                  router_hidden=None, balanced_routing=True, bias_lr=0.001):
        super().__init__()
        self.n_experts = n_experts
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.balanced_routing = balanced_routing
        self.bias_lr = bias_lr
        # Stacked expert parameters: (K, d_in|d_out, rank). Same init as individual
        # BilinearComposition instances (randn * init_scale).
        self.V = nn.Parameter(torch.randn(n_experts, d_in, rank) * init_scale)
        self.W = nn.Parameter(torch.randn(n_experts, d_in, rank) * init_scale)
        self.U = nn.Parameter(torch.randn(n_experts, d_out, rank) * init_scale)
        self.biases = nn.Parameter(torch.zeros(n_experts, d_out))
        rh = router_hidden or max(d_in, 32)
        self.router = nn.Sequential(
            nn.Linear(d_in * 2, rh),
            nn.GELU(),
            nn.Linear(rh, n_experts),
        )
        # Dynamic per-expert bias for balanced routing (buffer, no grad)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("usage_running", torch.zeros(n_experts))
        self.register_buffer("usage_count", torch.zeros(1))

    def forward(self, x, y):
        # x, y: (..., d_in). Output: (..., d_out).
        orig_shape = x.shape[:-1]
        N = x.numel() // self.d_in
        K, r, d_in, d_out = self.n_experts, self.rank, self.d_in, self.d_out

        x_flat = x.reshape(N, d_in)
        y_flat = y.reshape(N, d_in)

        # Router
        logits = self.router(torch.cat([x, y], dim=-1))  # (..., K)
        if self.balanced_routing:
            logits = logits + self.expert_bias
        weights = torch.softmax(logits, dim=-1)  # (..., K)
        weights_flat = weights.reshape(N, K)  # (N, K)

        if self.training and self.balanced_routing:
            with torch.no_grad():
                self.usage_running += weights_flat.sum(0).float()
                self.usage_count += float(N)

        # Merge experts along d_in axis: (d_in, K*r)
        V_m = self.V.permute(1, 0, 2).reshape(d_in, K * r)
        W_m = self.W.permute(1, 0, 2).reshape(d_in, K * r)
        xV = x_flat @ V_m  # (N, K*r)
        yW = y_flat @ W_m  # (N, K*r)
        prod = (xV * yW).reshape(N, K, r)

        # Apply per-token expert weights BEFORE U projection
        prod = prod * weights_flat.unsqueeze(-1)  # (N, K, r)
        prod_kr = prod.reshape(N, K * r)

        # Merge U along d_out axis: (K*r, d_out)
        U_m = self.U.permute(0, 2, 1).reshape(K * r, d_out)
        out = prod_kr @ U_m  # (N, d_out)

        # Weighted per-expert bias
        out = out + weights_flat @ self.biases  # (N, K) @ (K, d_out) = (N, d_out)
        return out.reshape(*orig_shape, d_out)

    @torch.no_grad()
    def update_bias(self):
        """Call after each step to adjust bias toward uniform expert usage.
        DeepSeek-V3 style: increase bias for under-used experts, decrease for over-used.
        """
        if not self.balanced_routing or self.usage_count.item() < 1:
            return
        avg_usage = self.usage_running / self.usage_count
        ideal = 1.0 / self.n_experts
        # If usage > ideal, expert is over-routed → decrease bias
        # If usage < ideal, expert is under-routed → increase bias
        delta = (ideal - avg_usage) * self.bias_lr
        self.expert_bias.add_(delta)
        # Reset stats
        self.usage_running.zero_()
        self.usage_count.zero_()

    def routing_entropy(self):
        """Diagnostic: returns mean entropy of expert routing distribution.
        Higher = more balanced (max ln(K)). Lower = router collapsed."""
        if self.usage_count.item() < 1:
            return float("nan")
        probs = self.usage_running / self.usage_count
        return float((-probs * torch.log(probs.clamp(min=1e-12))).sum())


class CausalMoBEBCNAttention(nn.Module):
    """Bi-BCN with Mixture of Bilinear Experts (MoBE) — O(L) cumsum-fused.

    Forward path: out_i = sum_{j<=i} compose(Q_i, K_j)
    Inverse path: out_i_inv = sum_{j<=i} compose(invK_j, invQ_i)
    Combined: out = forward + alpha_bi * inverse

    Bilinearity lets us factor the sum:
        sum_{j<=i} U_k ((V_k^T Q_i) ⊙ (W_k^T K_j))
      = U_k ((V_k^T Q_i) ⊙ cumsum_j(W_k^T K_j)_i)
    reducing O(B·L²·D·K) materialization to O(B·L·r·K).

    Routing is query-side only: w_k(Q_i). This is what lets the sum factor.
    Expert specialization is per-query, matching standard MoE paradigm.
    """
    def __init__(self, d_model, rank, n_experts=4, init_alpha=0.5, dropout=0.0,
                  balanced_routing=True, bias_lr=0.001, init_scale=0.02):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_experts = n_experts
        self.balanced_routing = balanced_routing
        self.bias_lr = bias_lr
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.W_inv = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.W_inv.weight.copy_(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        # Stacked expert parameters for fused einsum kernels
        self.V_fwd = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.W_fwd = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.U_fwd = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.b_fwd = nn.Parameter(torch.zeros(n_experts, d_model))
        self.V_inv = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.W_inv_exp = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.U_inv = nn.Parameter(torch.randn(n_experts, d_model, rank) * init_scale)
        self.b_inv = nn.Parameter(torch.zeros(n_experts, d_model))
        # Query-side router (single input, not pair)
        rh = max(d_model, 32)
        self.router = nn.Sequential(
            nn.Linear(d_model, rh),
            nn.GELU(),
            nn.Linear(rh, n_experts),
        )
        self.alpha_bi = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dropout = nn.Dropout(dropout)
        # DeepSeek-V3 style balanced-routing bias (no grad buffer)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("usage_running", torch.zeros(n_experts))
        self.register_buffer("usage_count", torch.zeros(1))

    # Kept for compatibility with update_mobe_biases() helper on the LM
    @property
    def compose(self):
        return self

    @torch.no_grad()
    def update_bias(self):
        if not self.balanced_routing or self.usage_count.item() < 1:
            return
        avg = self.usage_running / self.usage_count
        delta = (1.0 / self.n_experts - avg) * self.bias_lr
        self.expert_bias.add_(delta)
        self.usage_running.zero_()
        self.usage_count.zero_()

    def routing_entropy(self):
        if self.usage_count.item() < 1:
            return float("nan")
        p = self.usage_running / self.usage_count
        return float((-p * torch.log(p.clamp(min=1e-12))).sum())

    def _fused_branch(self, Q_side, K_side, V, W, U, b, attention_mask):
        """Compute sum_{j<=i} U_k ((V_k^T Q_i) ⊙ (W_k^T K_j)) via cumsum.

        Q_side: (B, T, D) — appears on "query" side of compose (carries i)
        K_side: (B, T, D) — appears on "key"   side of compose (carries j, summed)
        V, W, U: (K, D, r)
        b: (K, D) per-expert bias
        Returns: (B, T, D) attention output, already mean-normalized.
        """
        B, T, D = Q_side.shape
        K_exp = self.n_experts
        r = self.rank

        # Project Q, K into per-expert rank-r spaces
        # xV: (B, T, K, r)
        xV = torch.einsum('btd,kdr->btkr', Q_side, V)
        yW = torch.einsum('btd,kdr->btkr', K_side, W)

        # Apply key-side pad mask BEFORE cumsum so padded positions contribute 0
        if attention_mask is not None:
            pad = attention_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
            yW = yW * pad
            # Valid-count per position i: cumsum of mask along T
            n_valid = attention_mask.float().cumsum(dim=1).clamp(min=1.0)  # (B, T)
        else:
            n_valid = torch.arange(1, T + 1, device=Q_side.device, dtype=Q_side.dtype).unsqueeze(0).expand(B, T)

        # Causal cumulative sum of yW along T (j <= i)
        cum_yW = yW.cumsum(dim=1)  # (B, T, K, r)

        # Elementwise product — the bilinear composition, per expert per position
        prod = xV * cum_yW  # (B, T, K, r)

        # Router — query-side only
        logits = self.router(Q_side)  # (B, T, K)
        if self.balanced_routing:
            logits = logits + self.expert_bias
        weights = torch.softmax(logits, dim=-1)  # (B, T, K)

        if self.training and self.balanced_routing:
            with torch.no_grad():
                self.usage_running += weights.reshape(-1, K_exp).sum(0).float()
                self.usage_count += float(B * T)

        # Weight per-expert activations
        prod_w = prod * weights.unsqueeze(-1)  # (B, T, K, r)
        # Project to D via U (K, D, r)
        out = torch.einsum('btkr,kdr->btd', prod_w, U)  # (B, T, D)
        # Weighted per-expert bias
        out = out + weights @ b  # (B, T, K) @ (K, D) = (B, T, D)

        # Normalize by count of valid j's per i (mean over summed positions)
        out = out / n_valid.unsqueeze(-1)
        return out

    def forward(self, x, attention_mask=None):
        Q = self.W_Q(x)
        K = self.W_K(x)
        invQ = self.W_inv(Q)
        invK = self.W_inv(K)

        fwd = self._fused_branch(Q, K, self.V_fwd, self.W_fwd, self.U_fwd, self.b_fwd, attention_mask)
        # Inverse: sum_{j<=i} U_k ((V_k^T invK_j) ⊙ (W_k^T invQ_i))
        #        = U_k ((W_k^T invQ_i) ⊙ cumsum_j(V_k^T invK_j)_i)
        # So the "pointwise" side is invQ (uses W_inv_exp), "summed" side is invK (uses V_inv).
        # Pass them so V-arg handles pointwise and W-arg handles summed:
        inv = self._fused_branch(invQ, invK, self.W_inv_exp, self.V_inv, self.U_inv, self.b_inv, attention_mask)
        out = fwd + self.alpha_bi * inv
        return self.dropout(self.W_O(out))


class HybridCausalMoBEAttention(nn.Module):
    """MoBE-BCN + Causal MHA in parallel (best of both: algebraic + general)."""
    def __init__(self, d_model, n_heads, rank, n_experts=4, init_alpha=0.5, dropout=0.0):
        super().__init__()
        self.bcn_mobe = CausalMoBEBCNAttention(d_model, rank, n_experts=n_experts,
                                                  init_alpha=init_alpha, dropout=dropout)
        self.mha = CausalMHA(d_model, n_heads, dropout=dropout)

    def forward(self, x, attention_mask=None):
        return self.bcn_mobe(x, attention_mask) + self.mha(x, attention_mask)


class CausalBiBCNAttention(nn.Module):
    """Bi-BCN (inverse-aware) with causal masking.

    Forward path: out_i = sum_{j<=i} BCN(Q_i, K_j)
    Inverse path: out_i_inv = sum_{j<=i} BCN(W_inv K_j, W_inv Q_i)
        (swapped order mimics group property (g·h)^-1 = h^-1·g^-1)
    Combined: out = forward + alpha_bi * inverse

    Both paths share the same bilinear M (U/V/W). Only W_inv (learnable inverse
    projector) is added. Learnable alpha controls inverse contribution.
    """
    def __init__(self, d_model, rank, init_alpha=0.5, dropout=0.0):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        # Learnable inverse projector — initialized close to identity
        self.W_inv = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.W_inv.weight.copy_(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        # Bilinear primitive (shared between forward and inverse)
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.alpha_bi = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        invQ = self.W_inv(Q)
        invK = self.W_inv(K)

        Q_i = Q.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)
        K_j = K.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)
        invK_j = invK.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)
        invQ_i = invQ.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)

        # Forward: BCN(Q_i, K_j)
        fwd = ((Q_i @ self.V) * (K_j @ self.W)) @ self.U.T + self.bias
        # Inverse: BCN(invK_j, invQ_i) — note swapped order
        inv = ((invK_j @ self.V) * (invQ_i @ self.W)) @ self.U.T + self.bias

        out = (fwd + self.alpha_bi * inv).reshape(B, T, T, D)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        causal_f = causal.unsqueeze(0).unsqueeze(-1).float()
        out = out * causal_f

        if attention_mask is not None:
            pad_j = attention_mask.unsqueeze(1).unsqueeze(-1).float()
            out = out * pad_j
            n_valid = (causal.unsqueeze(0).float() * attention_mask.unsqueeze(1).float())
        else:
            n_valid = causal.unsqueeze(0).float().expand(B, T, T)
        n_valid_per_i = n_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = out.sum(dim=2) / n_valid_per_i

        return self.dropout(self.W_O(out))


class CausalBiTrilinearBCNAttention(nn.Module):
    """Combined: Bi-BCN forward + Bi-BCN inverse + Trilinear (with causal global).

    The most complete BCN variant — combines all three architectural innovations:
    - Bilinear forward (BCN base)
    - Bilinear inverse (Bi-BCN inverse-aware path)
    - Trilinear (Trilinear extension with causal context)

    Three learnable alphas: alpha_bi for inverse weight, alpha_tri for trilinear weight.
    """
    def __init__(self, d_model, rank, init_alpha_bi=0.5, init_alpha_tri=0.5, dropout=0.0):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.W_inv = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.W_inv.weight.copy_(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        # Bilinear primitive (shared forward/inverse)
        self.U_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W_b = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias_b = nn.Parameter(torch.zeros(d_model))
        # Trilinear primitive
        self.U_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.W_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.X_t = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.bias_t = nn.Parameter(torch.zeros(d_model))
        self.alpha_bi = nn.Parameter(torch.tensor(float(init_alpha_bi)))
        self.alpha_tri = nn.Parameter(torch.tensor(float(init_alpha_tri)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        invQ = self.W_inv(Q)
        invK = self.W_inv(K)

        # Causal cumulative mean for trilinear context
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            masked_x = x * mask_f
            cum_x = masked_x.cumsum(dim=1)
            cum_count = mask_f.cumsum(dim=1).clamp(min=1.0)
            x_global_causal = cum_x / cum_count
        else:
            cum_x = x.cumsum(dim=1)
            counts = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1, T, 1)
            x_global_causal = cum_x / counts

        Q_i = Q.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)
        K_j = K.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)
        invK_j = invK.unsqueeze(1).expand(B, T, T, D).reshape(-1, D)
        invQ_i = invQ.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)
        Z_i = x_global_causal.unsqueeze(2).expand(B, T, T, D).reshape(-1, D)

        # Forward bilinear: BCN(Q_i, K_j)
        fwd_bil = ((Q_i @ self.V_b) * (K_j @ self.W_b)) @ self.U_b.T + self.bias_b
        # Inverse bilinear: BCN(invK_j, invQ_i)
        inv_bil = ((invK_j @ self.V_b) * (invQ_i @ self.W_b)) @ self.U_b.T + self.bias_b
        # Trilinear: TriBCN(Q_i, K_j, Z_i)
        tri = ((Q_i @ self.V_t) * (K_j @ self.W_t) * (Z_i @ self.X_t)) @ self.U_t.T + self.bias_t

        out = (fwd_bil + self.alpha_bi * inv_bil + self.alpha_tri * tri).reshape(B, T, T, D)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        causal_f = causal.unsqueeze(0).unsqueeze(-1).float()
        out = out * causal_f

        if attention_mask is not None:
            pad_j = attention_mask.unsqueeze(1).unsqueeze(-1).float()
            out = out * pad_j
            n_valid = (causal.unsqueeze(0).float() * attention_mask.unsqueeze(1).float())
        else:
            n_valid = causal.unsqueeze(0).float().expand(B, T, T)
        n_valid_per_i = n_valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = out.sum(dim=2) / n_valid_per_i

        return self.dropout(self.W_O(out))


class HybridCausalBiAttention(nn.Module):
    """Bi-BCN + Causal MHA in parallel."""
    def __init__(self, d_model, n_heads, rank, init_alpha=0.5, dropout=0.0):
        super().__init__()
        self.bcn_bi = CausalBiBCNAttention(d_model, rank, init_alpha=init_alpha, dropout=dropout)
        self.mha = CausalMHA(d_model, n_heads, dropout=dropout)

    def forward(self, x, attention_mask=None):
        return self.bcn_bi(x, attention_mask) + self.mha(x, attention_mask)


class HybridCausalBiTrilinearAttention(nn.Module):
    """Bi-BCN + Trilinear + Causal MHA in parallel — full architectural stack."""
    def __init__(self, d_model, n_heads, rank, init_alpha_bi=0.5, init_alpha_tri=0.5, dropout=0.0):
        super().__init__()
        self.bcn_bi_tri = CausalBiTrilinearBCNAttention(d_model, rank,
                                                          init_alpha_bi=init_alpha_bi,
                                                          init_alpha_tri=init_alpha_tri,
                                                          dropout=dropout)
        self.mha = CausalMHA(d_model, n_heads, dropout=dropout)

    def forward(self, x, attention_mask=None):
        return self.bcn_bi_tri(x, attention_mask) + self.mha(x, attention_mask)


class KazdovBlock(nn.Module):
    """Pre-norm transformer-style block with hybrid causal attention.

    Flags (can be combined):
    - use_trilinear: adds trilinear path with causal global context
    - use_bi_bcn: adds inverse-aware BCN path with learnable W_inv
    - use_hybrid_mha (default True): adds standard MHA in PARALLEL with BCN paths.
        Set False to drop MHA — pure BCN, halves params + compute. Use this for
        the "BCN-only" architectural test (matches original KazdovLM-1M S5 setup
        which had no parallel MHA).

    Both flags on (use_bi_bcn + use_trilinear) = full BCN stack (with optional MHA).
    """
    def __init__(self, d_model, n_heads, rank, mlp_dim, dropout=0.0,
                 use_trilinear=False, use_bi_bcn=False, use_hybrid_mha=True,
                 use_mobe=False, n_experts=4,
                 init_alpha=0.5, init_alpha_bi=0.5, init_alpha_tri=0.5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        # MoBE branch (Mixture of Bilinear Experts) — takes precedence
        if use_mobe:
            if use_hybrid_mha:
                self.attn = HybridCausalMoBEAttention(
                    d_model, n_heads, rank, n_experts=n_experts,
                    init_alpha=init_alpha_bi, dropout=dropout)
            else:
                self.attn = CausalMoBEBCNAttention(
                    d_model, rank, n_experts=n_experts,
                    init_alpha=init_alpha_bi, dropout=dropout)
        elif use_bi_bcn and use_trilinear:
            if use_hybrid_mha:
                self.attn = HybridCausalBiTrilinearAttention(
                    d_model, n_heads, rank,
                    init_alpha_bi=init_alpha_bi, init_alpha_tri=init_alpha_tri,
                    dropout=dropout
                )
            else:
                # Pure BCN: no parallel MHA. Same compute as just CausalBiTrilinearBCNAttention.
                self.attn = CausalBiTrilinearBCNAttention(
                    d_model, rank,
                    init_alpha_bi=init_alpha_bi, init_alpha_tri=init_alpha_tri,
                    dropout=dropout
                )
        elif use_bi_bcn:
            if use_hybrid_mha:
                self.attn = HybridCausalBiAttention(d_model, n_heads, rank,
                                                     init_alpha=init_alpha_bi, dropout=dropout)
            else:
                self.attn = CausalBiBCNAttention(d_model, rank,
                                                  init_alpha=init_alpha_bi, dropout=dropout)
        elif use_trilinear:
            if use_hybrid_mha:
                self.attn = HybridCausalTrilinearAttention(d_model, n_heads, rank,
                                                            init_alpha=init_alpha_tri, dropout=dropout)
            else:
                self.attn = CausalTrilinearBCNAttention(d_model, rank,
                                                        init_alpha=init_alpha_tri, dropout=dropout)
        else:
            if use_hybrid_mha:
                self.attn = HybridCausalAttention(d_model, n_heads, rank, dropout=dropout)
            else:
                self.attn = CausalBCNAttention(d_model, rank, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class KazdovLM(nn.Module):
    """Autoregressive Kazdov language model.

    Standard transformer-style stack with:
    - Token embedding + learned positional embedding
    - N hybrid Kazdov blocks (causal BCN + causal MHA)
    - Final LayerNorm + LM head (tied to embedding optional)

    Use forward(input_ids, labels) for training loss.
    Use generate(input_ids, max_new_tokens) for inference.
    """
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6, rank=64,
                 mlp_dim=None, max_len=512, dropout=0.0, tie_embeddings=True,
                 use_trilinear=False, use_bi_bcn=False, use_hybrid_mha=True,
                 use_mobe=False, n_experts=4,
                 init_alpha=0.5, init_alpha_bi=0.5, init_alpha_tri=0.5):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 4 * d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_trilinear = use_trilinear
        self.use_bi_bcn = use_bi_bcn
        self.use_hybrid_mha = use_hybrid_mha
        self.use_mobe = use_mobe
        self.n_experts = n_experts if use_mobe else 1

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            KazdovBlock(d_model, n_heads, rank, mlp_dim, dropout,
                          use_trilinear=use_trilinear, use_bi_bcn=use_bi_bcn,
                          use_hybrid_mha=use_hybrid_mha,
                          use_mobe=use_mobe, n_experts=n_experts,
                          init_alpha=init_alpha, init_alpha_bi=init_alpha_bi,
                          init_alpha_tri=init_alpha_tri)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids: (B, T)
        attention_mask: (B, T) bool, True for real tokens
        labels: (B, T) for training (-100 ignored, shifted internally)

        Returns dict with 'logits' (B, T, V) and 'loss' if labels provided.
        """
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"input length {T} exceeds max_len {self.max_len}")
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.embed(input_ids) + self.pos(pos_ids)
        h = self.drop(h)
        for block in self.blocks:
            if getattr(self, "_grad_checkpoint", False) and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, attention_mask, use_reentrant=False)
            else:
                h = block(h, attention_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)

        out = {"logits": logits}
        if labels is not None:
            # Shift: predict t+1 from t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    def enable_grad_checkpointing(self):
        """Enable gradient checkpointing on transformer blocks (saves memory at compute cost)."""
        self._grad_checkpoint = True

    def update_mobe_biases(self):
        """Update MoBE expert biases (DeepSeek-V3 bias-balanced gating).
        Call after each training step to dynamically rebalance expert usage."""
        for module in self.modules():
            if isinstance(module, (MixtureBilinear, CausalMoBEBCNAttention)):
                module.update_bias()

    def routing_diagnostics(self):
        """Returns dict per layer with routing entropy + expert usage."""
        diag = {}
        for i, block in enumerate(self.blocks):
            for module in block.modules():
                if isinstance(module, (MixtureBilinear, CausalMoBEBCNAttention)):
                    if module.usage_count.item() > 0:
                        usage = (module.usage_running / module.usage_count).cpu().tolist()
                        diag[f"block{i}.expert_usage"] = usage
                        diag[f"block{i}.routing_entropy"] = module.routing_entropy()
        return diag

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=1.0, top_k=None,
                 eos_token_id=None, attention_mask=None):
        """Greedy / sampling generation. Returns (B, T+new) tensor."""
        self.eval()
        B = input_ids.shape[0]
        for _ in range(max_new_tokens):
            T = input_ids.shape[1]
            if T >= self.max_len:
                break
            current_mask = attention_mask
            if current_mask is None:
                current_mask = torch.ones_like(input_ids, dtype=torch.bool)
            out = self.forward(input_ids, attention_mask=current_mask)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                top_vals, top_idx = logits.topk(top_k, dim=-1)
                mask = torch.zeros_like(logits).scatter_(-1, top_idx, 1.0).bool()
                logits = logits.masked_fill(~mask, float('-inf'))
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask,
                                              torch.ones((B, 1), dtype=torch.bool, device=input_ids.device)], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== KazdovLM sanity ===")
    vocab_size = 1000
    model = KazdovLM(vocab_size, d_model=128, n_layers=2, n_heads=4, rank=32, max_len=64)
    print(f"params: {count_params(model):,}")

    # Forward
    x = torch.randint(0, vocab_size, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 12:] = False
    labels = x.clone()
    labels[~mask] = -100
    out = model(x, attention_mask=mask, labels=labels)
    print(f"logits: {out['logits'].shape}, loss: {out['loss'].item():.4f}")

    # Generate
    prompt = torch.tensor([[1, 2, 3, 4]])
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.5, top_k=10)
    print(f"generated: {generated.shape} = {generated.tolist()}")
    print("Sanity passed.")
