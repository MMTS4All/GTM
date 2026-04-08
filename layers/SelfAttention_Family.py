"""
Self-attention mechanisms for transformer models.

This module implements various attention mechanisms used in transformer architectures,
including full attention and probabilistic attention for efficient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """
    Standard scaled dot-product attention mechanism.
    
    This class implements the full attention mechanism as described in the
    original transformer paper, with optional masking and attention dropout.
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initialize the FullAttention module.
        
        Args:
            mask_flag (bool): Whether to apply masking. Defaults to True.
            factor (int): Factor parameter (unused in this implementation). Defaults to 5.
            scale (float, optional): Scaling factor for attention scores. Defaults to None.
            attention_dropout (float): Dropout probability for attention weights. Defaults to 0.1.
            output_attention (bool): Whether to output attention weights. Defaults to False.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        """
        Apply full attention to input queries, keys, and values.
        
        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, H, E].
            keys (torch.Tensor): Key tensor of shape [B, S, H, E].
            values (torch.Tensor): Value tensor of shape [B, S, H, D].
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            
        Returns:
            tuple: (output, attention_weights) where:
                - output: Tensor of shape [B, L, H, D]
                - attention_weights: Attention weights tensor or None
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply masking if required
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Apply softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A = torch.nan_to_num(A, nan=0.0)
        
        # Compute weighted values
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    Probabilistic attention mechanism for efficient computation.
    
    This class implements a sparse attention mechanism that reduces
    computational complexity by sampling key-query interactions.
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initialize the ProbAttention module.
        
        Args:
            mask_flag (bool): Whether to apply masking. Defaults to True.
            factor (int): Factor for sampling keys and queries. Defaults to 5.
            scale (float, optional): Scaling factor for attention scores. Defaults to None.
            attention_dropout (float): Dropout probability for attention weights. Defaults to 0.1.
            output_attention (bool): Whether to output attention weights. Defaults to False.
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Sample key-query interactions for efficient attention computation.
        
        Args:
            Q (torch.Tensor): Query tensor of shape [B, H, L, D].
            K (torch.Tensor): Key tensor of shape [B, H, L_K, D].
            sample_k (int): Number of keys to sample.
            n_top (int): Number of top queries to select.
            
        Returns:
            tuple: (Q_K_scores, top_indices) where:
                - Q_K_scores: Attention scores for sampled interactions
                - top_indices: Indices of top queries
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Get initial context for attention computation.
        
        Args:
            V (torch.Tensor): Value tensor of shape [B, H, L_V, D].
            L_Q (int): Length of queries.
            
        Returns:
            torch.Tensor: Initial context tensor.
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Update context with selected top-k queries.
        
        Args:
            context_in (torch.Tensor): Initial context tensor.
            V (torch.Tensor): Value tensor.
            scores (torch.Tensor): Attention scores.
            index (torch.Tensor): Indices of top queries.
            L_Q (int): Length of queries.
            attn_mask (torch.Tensor): Attention mask.
            
        Returns:
            tuple: (updated_context, attention_weights) where:
                - updated_context: Updated context tensor
                - attention_weights: Attention weights or None
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        """
        Apply probabilistic attention to input queries, keys, and values.
        
        Args:
            queries (torch.Tensor): Query tensor of shape [B, L_Q, H, D].
            keys (torch.Tensor): Key tensor of shape [B, L_K, H, D].
            values (torch.Tensor): Value tensor of shape [B, L_V, H, D].
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            
        Returns:
            tuple: (output, attention_weights) where:
                - output: Tensor of shape [B, L_Q, H, D]
                - attention_weights: Attention weights tensor or None
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # Get the context
        context = self._get_initial_context(values, L_Q)
        # Update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention layer that wraps attention mechanisms with linear projections.
    
    This class applies linear projections to input tensors before passing them
    to the inner attention mechanism, and applies a final linear projection
    to the output.
    """
    
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        Initialize the AttentionLayer.
        
        Args:
            attention (nn.Module): Inner attention mechanism.
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_keys (int, optional): Dimension of keys. Defaults to d_model // n_heads.
            d_values (int, optional): Dimension of values. Defaults to d_model // n_heads.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        Apply attention layer to input queries, keys, and values.
        
        Args:
            queries (torch.Tensor): Query tensor of shape [B, L, d_model].
            keys (torch.Tensor): Key tensor of shape [B, S, d_model].
            values (torch.Tensor): Value tensor of shape [B, S, d_model].
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            
        Returns:
            tuple: (output, attention_weights) where:
                - output: Tensor of shape [B, L, d_model]
                - attention_weights: Attention weights from inner attention
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        if attn_mask is not None:
            attn_mask = ~attn_mask.unsqueeze(1).repeat(1,H,1,1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
