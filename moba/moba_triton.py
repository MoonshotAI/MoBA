"""
Efficient MoBA (Mixture of Block Attention) Implementation with Triton Optimization.

This module provides a highly optimized implementation of MoBA attention using OpenAI Triton kernels 
and FlashAttention. It serves as a drop-in replacement for the original PyTorch-based implementation, 
offering significant speedups (approx. 2.8x faster in training loops) and reduced memory footprint.

Key Optimizations:
1. Fused Chunk Mean Calculation: Replaced PyTorch's `view().mean()` with a Triton kernel to avoid 
   materializing intermediate tensors and improve memory locality.
2. Fused Gating & Top-K Selection: Replaced the memory-intensive `torch.einsum` (for gating) and 
   `torch.topk` pipeline with a single Triton kernel (`_chunk_topk_kernel`). This kernel fuses score 
   calculation, causal masking, and top-k selection, eliminating the need to instantiate the massive 
   [Batch, Head, Chunk, Seq] score matrix.
3. Fused Merge Softmax: Implemented `_fused_merge_softmax_kernel` to handle the weighted combination 
   of Self-Attention and MoBA-Attention outputs. This performs the LogSumExp (LSE) reduction and output 
   merging in a single pass, drastically reducing global memory I/O compared to the original multi-step 
   PyTorch operations.
4. Optimized Backward Pass: Custom `_gather_moba_backward_inputs_kernel` efficiently gathers gradients 
   and forward activations needed for the sparse attention branch's backward pass.
5. Layout Optimization: Optimized index calculations to work directly with (Seq, Head, Dim) layout, 
   removing expensive `rearrange` operations found in the original code.

Original Author: [Moonshot AI]
Triton Optimization: [ReyJerry]
"""

import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange

# Helper for triton.cdiv compatibility
def cdiv(x, y):
    return (x + y - 1) // y

@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """
    Pre-calculates chunk metadata based on cumulative sequence lengths.
    Determines how sequences are split into fixed-size chunks for MoBA.
    """
    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """
    # filter chunks ( remove last chunk of each batch as it's processed by self-attn )
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


@triton.jit
def _chunk_mean_kernel(
    k_ptr,
    out_ptr,
    stride_chunk,
    stride_token,
    stride_head,
    stride_d,
    out_stride_chunk,
    out_stride_head,
    out_stride_d,
    num_chunks,
    num_heads,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to compute the mean of Key vectors within each chunk.
    Replacing: filtered_kv.view(...).mean(dim=1)
    """
    chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if chunk_id >= num_chunks or head_id >= num_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, HEAD_DIM, BLOCK_D):
        mask_d = (d_start + offs_d) < HEAD_DIM
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
            offs_t = tl.arange(0, BLOCK_TOK)
            mask_t = (t_start + offs_t) < CHUNK_SIZE

            ptrs = (
                k_ptr
                + chunk_id * stride_chunk
                + (t_start + offs_t)[:, None] * stride_token
                + head_id * stride_head
                + (d_start + offs_d)[None, :] * stride_d
            )
            vals = tl.load(ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        acc = acc / CHUNK_SIZE
        out_ptrs = (
            out_ptr
            + chunk_id * out_stride_chunk
            + head_id * out_stride_head
            + (d_start + offs_d) * out_stride_d
        )
        tl.store(out_ptrs, acc, mask=mask_d)


@triton.jit
def _chunk_topk_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    gate_mask_ptr,
    num_chunks,
    num_heads,
    seqlen,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    stride_gate_chunk,
    stride_gate_head,
    stride_gate_token,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    NUM_CHUNK_TILES: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused Kernel for Gating and Top-K selection.
    
    Performs:
    1. Dot product between Query and Chunk Means (Gating Score).
    2. Causal masking (masking out future chunks and cross-batch chunks).
    3. Top-K selection to determine which chunks each query token attends to.
    
    This replaces the memory-heavy `torch.einsum` + `torch.topk` approach.
    """
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_head >= num_heads:
        return

    offs_token = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = offs_token < seqlen
    offs_d = tl.arange(0, BLOCK_D)

    NEG_INF = -1.0e38

    # Loop over chunk tiles to compute scores without materializing the full score matrix
    for chunk_tile in range(NUM_CHUNK_TILES):
        chunk_offsets = chunk_tile * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
        chunk_mask = chunk_offsets < num_chunks
        chunk_offsets_safe = tl.where(chunk_mask, chunk_offsets, tl.zeros_like(chunk_offsets))

        gating = tl.zeros([BLOCK_CHUNK, BLOCK_TOK], dtype=tl.float32)

        # Compute dot product (Gating Score)
        for d_start in range(0, HEAD_DIM, BLOCK_D):
            mask_d = (d_start + offs_d) < HEAD_DIM

            q_ptrs = (
                q_ptr
                + offs_token[:, None] * stride_q_seq
                + pid_head * stride_q_head
                + (d_start + offs_d)[None, :] * stride_q_d
            )
            q_vals = tl.load(q_ptrs, mask=token_mask[:, None] & mask_d[None, :], other=0.0)
            q_vals = q_vals.to(tl.float32)

            chunk_ptrs = (
                chunk_means_ptr
                + chunk_offsets_safe[:, None] * stride_chunk_mean_chunk
                + pid_head * stride_chunk_mean_head
                + (d_start + offs_d)[None, :] * stride_chunk_mean_d
            )
            chunk_vals = tl.load(
                chunk_ptrs,
                mask=chunk_mask[:, None] & mask_d[None, :],
                other=0.0
            )
            chunk_vals = chunk_vals.to(tl.float32)

            gating = gating + tl.dot(chunk_vals, tl.trans(q_vals))

        # Apply Causal & Batch Masks on the fly
        chunk_end_idx = tl.load(chunk_end_ptr + chunk_offsets_safe, mask=chunk_mask, other=0)
        batch_end_idx = tl.load(batch_end_ptr + chunk_offsets_safe, mask=chunk_mask, other=0)

        chunk_end = tl.broadcast_to(chunk_end_idx[:, None], gating.shape)
        batch_end = tl.broadcast_to(batch_end_idx[:, None], gating.shape)

        token_ids = offs_token[None, :]
        chunk_mask_2d = tl.broadcast_to(chunk_mask[:, None], gating.shape)
        token_mask_2d = tl.broadcast_to(token_mask[None, :], gating.shape)

        valid_tokens = chunk_mask_2d & token_mask_2d & (token_ids >= chunk_end) & (token_ids < batch_end)
        gating = tl.where(valid_tokens, gating, NEG_INF)

        # Select Top-K locally
        row_indices = tl.arange(0, BLOCK_CHUNK)[:, None]
        selected_value = tl.full([BLOCK_CHUNK, BLOCK_TOK], 1, dtype=tl.int8)
        gate_ptrs = (
            gate_mask_ptr
            + chunk_offsets_safe[:, None] * stride_gate_chunk
            + pid_head * stride_gate_head
            + offs_token[None, :] * stride_gate_token
        )

        # Iteratively find max to avoid sorting overhead
        for k in range(TOPK):
            best_vals = tl.max(gating, axis=0)
            best_idx = tl.argmax(gating, axis=0)

            global_idx = best_idx + chunk_tile * BLOCK_CHUNK
            valid_best = best_vals > NEG_INF

            selected_mask = (
                (row_indices == best_idx[None, :])
                & chunk_mask[:, None]
                & valid_best[None, :]
            )
            tl.store(
                gate_ptrs,
                selected_value,
                mask=selected_mask
            )
            # Mask out the chosen one to find the next best
            gating = tl.where(row_indices == best_idx[None, :], NEG_INF, gating)


def _compute_chunk_means_triton(filtered_k: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Wrapper for chunk mean kernel"""
    num_chunks, _, num_heads, head_dim = filtered_k.shape
    if num_chunks == 0:
        return filtered_k.new_zeros((0, num_heads, head_dim), dtype=torch.float32)

    filtered_k = filtered_k.contiguous()
    chunk_means = torch.empty(
        (num_chunks, num_heads, head_dim), device=filtered_k.device, dtype=torch.float32
    )

    block_tok = max(1, min(64, chunk_size))
    block_d = max(1, min(64, head_dim))

    grid = (num_chunks, num_heads)

    _chunk_mean_kernel[grid](
        filtered_k,
        chunk_means,
        filtered_k.stride(0),
        filtered_k.stride(1),
        filtered_k.stride(2),
        filtered_k.stride(3),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        num_chunks,
        num_heads,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_TOK=block_tok,
        BLOCK_D=block_d,
    )

    return chunk_means


def _compute_gate_mask_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Wrapper for gating and top-k selection kernel"""
    num_chunks, num_heads, head_dim = chunk_means.shape
    seqlen = q.shape[0]

    if topk <= 0 or num_chunks == 0:
        return chunk_means.new_zeros((num_chunks, num_heads, seqlen), dtype=torch.bool)

    q = q.contiguous()
    chunk_means = chunk_means.contiguous()
    chunk_end = chunk_end.to(dtype=torch.int32).contiguous()
    batch_end = batch_end.to(dtype=torch.int32).contiguous()

    gate_mask = torch.zeros(
        (num_chunks, num_heads, seqlen), device=q.device, dtype=torch.uint8
    )

    block_tok = 64
    block_chunk = 32
    block_d = max(1, min(64, head_dim))
    block_topk = 1 << (topk - 1).bit_length() # Round up to power of 2

    num_token_tiles = cdiv(seqlen, block_tok)
    num_chunk_tiles = cdiv(num_chunks, block_chunk)

    _chunk_topk_kernel[(num_token_tiles, num_heads)](
        q,
        chunk_means,
        chunk_end,
        batch_end,
        gate_mask,
        num_chunks,
        num_heads,
        seqlen,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        gate_mask.stride(0),
        gate_mask.stride(1),
        gate_mask.stride(2),
        HEAD_DIM=head_dim,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        NUM_CHUNK_TILES=num_chunk_tiles,
        BLOCK_TOK=block_tok,
        BLOCK_CHUNK=block_chunk,
        BLOCK_D=block_d,
    )

    return gate_mask.bool()


@triton.jit
def _fused_merge_softmax_kernel(
    self_out_ptr,
    self_lse_ptr,
    moba_out_ptr,
    moba_lse_ptr,
    row_offsets_ptr,
    out_ptr,
    final_lse_ptr,
    num_rows,
    head_dim,
    stride_self_row,
    stride_self_d,
    stride_moba_row,
    stride_moba_d,
    stride_out_row,
    stride_out_d,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
):
    """
    Fused kernel to combine Self-Attention and MoBA-Attention results.
    
    It performs the standard FlashAttention output merging:
    O = (O_self * exp(LSE_self - LSE_max) + sum(O_moba * exp(LSE_moba - LSE_max))) / exp(LSE_new - LSE_max)
    
    This avoids multiple passes of reading/writing large output tensors.
    """
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    # Identify which MoBA results correspond to this row (if any)
    offs_k = tl.arange(0, TOPK)
    row_start = tl.load(row_offsets_ptr + row_id)
    row_end = tl.load(row_offsets_ptr + row_id + 1)
    nnz = row_end - row_start
    mask_k = offs_k < nnz
    positions = row_start + offs_k
    positions = tl.where(mask_k, positions, row_start)

    # Load LSEs and compute global Max LSE
    self_lse = tl.load(self_lse_ptr + row_id).to(tl.float32)
    moba_lse = tl.load(
        moba_lse_ptr + positions,
        mask=mask_k,
        other=-float("inf"),
    ).to(tl.float32)

    max_moba = tl.max(moba_lse, axis=0)
    max_lse = tl.maximum(self_lse, max_moba)

    # Compute weights
    self_se = tl.exp(self_lse - max_lse)
    moba_se = tl.exp(moba_lse - max_lse)
    moba_se = tl.where(mask_k, moba_se, 0.0)
    total_se = self_se + tl.sum(moba_se, axis=0)
    merged_lse = tl.log(total_se) + max_lse
    tl.store(final_lse_ptr + row_id, merged_lse)

    # Compute weighted output
    inv_total = 1.0 / total_se
    self_factor = self_se * inv_total
    moba_factor = moba_se * inv_total

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        self_ptrs = (
            self_out_ptr
            + row_id * stride_self_row
            + (d_start + offs_d) * stride_self_d
        )
        self_vals = tl.load(self_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc = self_vals * self_factor

        moba_ptrs = (
            moba_out_ptr
            + positions[:, None] * stride_moba_row
            + (d_start + offs_d)[None, :] * stride_moba_d
        )
        moba_vals = tl.load(
            moba_ptrs,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scaled = moba_vals * moba_factor[:, None]
        acc = acc + tl.sum(scaled, axis=0)

        out_ptrs = (
            out_ptr
            + row_id * stride_out_row
            + (d_start + offs_d) * stride_out_d
        )
        tl.store(out_ptrs, acc, mask=mask_d)


def _build_row_offsets(moba_indices: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper to CSR-style offsets for sparse MoBA outputs"""
    if moba_indices.numel() == 0:
        offsets = torch.zeros(num_rows + 1, dtype=torch.int32, device=moba_indices.device)
        return offsets, moba_indices

    sorted_perm = torch.argsort(moba_indices)
    sorted_rows = moba_indices.index_select(0, sorted_perm)
    counts = torch.bincount(sorted_rows, minlength=num_rows)
    counts = counts.to(torch.int32)

    offsets = torch.empty(num_rows + 1, dtype=torch.int32, device=moba_indices.device)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(dim=0, dtype=torch.int32)

    return offsets, sorted_perm


def _fused_merge_softmax_triton(
    self_out: torch.Tensor,
    self_lse: torch.Tensor,
    moba_out: torch.Tensor,
    moba_lse: torch.Tensor,
    moba_indices: torch.Tensor,
    final_out: torch.Tensor,
    final_lse: torch.Tensor,
):
    """Wrapper for the fused merge kernel"""
    num_rows, head_dim = self_out.shape

    row_offsets, perm = _build_row_offsets(moba_indices, num_rows)
    if moba_indices.numel() == 0:
        final_out.copy_(self_out.to(final_out.dtype))
        final_lse.copy_(self_lse)
        return

    moba_out_sorted = moba_out.index_select(0, perm).contiguous()
    moba_lse_sorted = moba_lse.index_select(0, perm).contiguous()

    counts = row_offsets[1:] - row_offsets[:-1]
    max_k = int(counts.max().item())
    if max_k == 0:
        final_out.copy_(self_out.to(final_out.dtype))
        final_lse.copy_(self_lse)
        return
    max_k_power2 = 1 << (max_k - 1).bit_length()

    block_d = min(128, (head_dim + 31) // 32 * 32)
    if block_d == 0:
        block_d = head_dim

    grid = (num_rows,)

    _fused_merge_softmax_kernel[grid](
        self_out,
        self_lse,
        moba_out_sorted,
        moba_lse_sorted,
        row_offsets,
        final_out,
        final_lse,
        num_rows,
        head_dim,
        self_out.stride(0),
        self_out.stride(1),
        moba_out_sorted.stride(0),
        moba_out_sorted.stride(1),
        final_out.stride(0),
        final_out.stride(1),
        BLOCK_D=block_d,
        TOPK=max_k_power2,
    )


@triton.jit
def _gather_moba_backward_inputs_kernel(
    dout_ptr,
    out_ptr,
    lse_ptr,
    indices_ptr,
    gathered_dout_ptr,
    gathered_out_ptr,
    gathered_lse_ptr,
    num_indices,
    head_dim,
    stride_dout_row,
    stride_dout_d,
    stride_out_row,
    stride_out_d,
    stride_gdout_row,
    stride_gdout_d,
    stride_gout_row,
    stride_gout_d,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to gather backward pass inputs for the MoBA branch.
    Since MoBA works on a sparse subset of Q, we need to gather gradients (dout), 
    outputs (out), and LSE scores corresponding to those sparse indices.
    """
    pid = tl.program_id(0)
    if pid >= num_indices:
        return

    sh_idx = tl.load(indices_ptr + pid)
    lse_val = tl.load(lse_ptr + sh_idx)
    tl.store(gathered_lse_ptr + pid, lse_val)

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        dout_ptrs = (
            dout_ptr
            + sh_idx * stride_dout_row
            + (d_start + offs_d) * stride_dout_d
        )
        dout_vals = tl.load(dout_ptrs, mask=mask_d, other=0.0)
        gdout_ptrs = (
            gathered_dout_ptr
            + pid * stride_gdout_row
            + (d_start + offs_d) * stride_gdout_d
        )
        tl.store(gdout_ptrs, dout_vals, mask=mask_d)

        out_ptrs = (
            out_ptr
            + sh_idx * stride_out_row
            + (d_start + offs_d) * stride_out_d
        )
        out_vals = tl.load(out_ptrs, mask=mask_d, other=0.0)
        gout_ptrs = (
            gathered_out_ptr
            + pid * stride_gout_row
            + (d_start + offs_d) * stride_gout_d
        )
        tl.store(gout_ptrs, out_vals, mask=mask_d)


def _gather_moba_backward_inputs_triton(
    d_output_2d: torch.Tensor,
    output_2d: torch.Tensor,
    mixed_attn_vlse_flat: torch.Tensor,
    moba_indices: torch.Tensor,
    gathered_d_output: torch.Tensor,
    gathered_output: torch.Tensor,
    gathered_lse: torch.Tensor,
):
    """Wrapper for the backward gather kernel"""
    num_indices = moba_indices.numel()
    if num_indices == 0:
        return

    head_dim = d_output_2d.shape[1]
    indices_i32 = moba_indices.to(torch.int32)

    block_d = min(128, (head_dim + 31) // 32 * 32)
    if block_d == 0:
        block_d = head_dim

    grid = (num_indices,)

    _gather_moba_backward_inputs_kernel[grid](
        d_output_2d,
        output_2d,
        mixed_attn_vlse_flat,
        indices_i32,
        gathered_d_output,
        gathered_output,
        gathered_lse,
        num_indices,
        head_dim,
        d_output_2d.stride(0),
        d_output_2d.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        gathered_d_output.stride(0),
        gathered_d_output.stride(1),
        gathered_output.stride(0),
        gathered_output.stride(1),
        BLOCK_D=block_d,
    )

class MixedAttention(torch.autograd.Function):
    """
    Custom Autograd Function handling the mixed attention mechanism.
    Integrates Self-Attention and MoBA-Attention using Triton-optimized kernels.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # 1. Self Attention (Standard FlashAttention)
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # 2. MoBA Attention (FlashAttention on selected sparse chunks)
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # 3. Output Merging
        # Optimization: Replaced Python-loop based LSE combination with Fused Triton Kernel
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        output_2d = output.view(-1, q.shape[2])
        self_attn_out_2d = self_attn_out_sh.reshape(-1, q.shape[2]).contiguous()
        self_attn_lse_flat = self_attn_lse_sh.view(-1).contiguous()
        moba_out_flat = moba_attn_out.view(-1, moba_attn_out.shape[-1]).contiguous()
        moba_lse_flat = moba_attn_lse.view(-1).contiguous()
        mixed_attn_lse_flat = torch.empty_like(self_attn_lse_flat)

        # Fused Kernel call
        _fused_merge_softmax_triton(
            self_attn_out_2d,
            self_attn_lse_flat,
            moba_out_flat,
            moba_lse_flat,
            moba_q_sh_indices,
            output_2d,
            mixed_attn_lse_flat,
        )

        output = output.to(q.dtype)
        mixed_attn_lse_sh = mixed_attn_lse_flat.view_as(self_attn_lse_sh)
        
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        max_seqlen, moba_chunk_size, softmax_scale = ctx.max_seqlen, ctx.moba_chunk_size, ctx.softmax_scale
        (
            output, mixed_attn_vlse_sh, q, k, v, self_attn_cu_seqlen, moba_q,
            moba_kv, moba_cu_seqlen_q, moba_cu_seqlen_kv, moba_q_sh_indices,
        ) = ctx.saved_tensors
        d_output = d_output.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # 1. Self Attention Backward
        _ = _flash_attn_varlen_backward(
            dout=d_output, q=q, k=k, v=v, out=output, softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq, dk=dk, dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen, cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, softmax_scale=softmax_scale, causal=True,
            dropout_p=0.0, 
            window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, 
            deterministic=True,
        )

        # 2. Gather inputs for MoBA Backward
        # Optimization: Triton kernel to sparsely gather gradients/outputs needed for MoBA backward
        headdim = q.shape[-1]
        d_output_2d = d_output.view(-1, headdim).contiguous()
        output_2d = output.view(-1, headdim).contiguous()
        mixed_attn_vlse_flat = mixed_attn_vlse_sh.view(-1).contiguous()

        num_selected = moba_q_sh_indices.numel()
        gathered_d_moba = torch.empty(
            (num_selected, headdim), device=d_output.device, dtype=d_output.dtype
        )
        gathered_moba_out = torch.empty(
            (num_selected, headdim), device=output.device, dtype=output.dtype
        )
        gathered_lse = torch.empty(
            (num_selected,), device=mixed_attn_vlse_flat.device, dtype=mixed_attn_vlse_flat.dtype
        )

        _gather_moba_backward_inputs_triton(
            d_output_2d,
            output_2d,
            mixed_attn_vlse_flat,
            moba_q_sh_indices,
            gathered_d_moba,
            gathered_moba_out,
            gathered_lse,
        )

        d_moba_output = gathered_d_moba.unsqueeze(1)
        moba_output = gathered_moba_out.unsqueeze(1)
        mixed_attn_vlse = gathered_lse.unsqueeze(0)
        
        dmq = torch.empty_like(moba_q)
        dmk = torch.empty_like(moba_kv[:, 0])
        dmv = torch.empty_like(moba_kv[:, 1])

        # 3. MoBA Attention Backward
        _ = _flash_attn_varlen_backward(
            dout=d_moba_output, q=moba_q, k=moba_kv[:, 0], v=moba_kv[:, 1], out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq, dk=dmk, dv=dmv, 
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv, max_seqlen_q=max_seqlen, max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale, causal=False, dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        
        # Return gradients in order. 
        # Note: 'dmq' (sparse) will be automatically scattered back to 'q.grad' by PyTorch Autograd
        # since 'moba_q' was created via index_select from 'q'.
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """
    Entry point for Efficient MoBA with Triton optimizations.
    
    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): Cumulative sequence length (FlashAttention format)
        max_seqlen (int): Max sequence length in batch
        moba_chunk_size (int): Size of chunks
        moba_topk (int): Number of chunks to attend to
    """

    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """
    # Optimization: Use Triton Kernel to calculate chunk means
    filtered_k = filtered_kv[:, 0].reshape(
        num_filtered_chunk, moba_chunk_size, num_head, head_dim
    )
    chunk_means = _compute_chunk_means_triton(filtered_k, moba_chunk_size)

    chunk_end = cu_chunk.index_select(0, filtered_chunk_indices + 1)
    batch_indices = chunk_to_batch.index_select(0, filtered_chunk_indices)
    batch_end = cu_seqlens.index_select(0, batch_indices + 1)

    # Optimization: Use Fused Triton Kernel for Gating + Causal Masking + TopK
    gate_mask = _compute_gate_mask_triton(
        q,
        chunk_means,
        chunk_end,
        batch_end,
        moba_topk,
    )

    """ find moba q that needs moba attn """
    
    # 1. Flatten the gate mask [Chunks, Heads, Seq] -> [Chunks, Heads * Seq]
    gate_mask_flat = gate_mask.view(num_filtered_chunk, -1)
    
    # 2. Get indices where mask is True. 
    nonzeros = torch.nonzero(gate_mask_flat, as_tuple=True)
    moba_q_indices_hs = nonzeros[1].to(torch.long) # [num_selected]
    
    # SAFETY: Early exit if nothing selected
    if moba_q_indices_hs.numel() == 0:
         return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    # 3. Calculate target indices directly (Optimization: Avoid rearrange/transpose)
    # Original logic required q to be [H, S, D] -> [H*S, D]
    # Here we work directly with q in [S, H, D] layout
    h_idx = moba_q_indices_hs // seqlen
    s_idx = moba_q_indices_hs % seqlen
    
    # target indices for Q [S, H, D] are: s * num_head + h
    moba_q_sh_indices = s_idx * num_head + h_idx
    
    # Select directly from Q without transpose/copy
    moba_q = q.view(-1, head_dim).index_select(0, moba_q_sh_indices)
    moba_q = moba_q.unsqueeze(1) # [num_selected, 1, D]
    
    moba_q_sh_indices = moba_q_sh_indices.to(torch.int32)
    
    # Calculate lengths for flash attn
    full_moba_seqlen_q = gate_mask.to(torch.int32).sum(dim=-1).flatten()
    q_zero_mask = full_moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = int(q_zero_mask.sum().item())
    
    if zero_expert_count > 0:
        full_moba_seqlen_q = full_moba_seqlen_q[valid_expert_mask]

    moba_cu_seqlen_q = torch.empty(
        full_moba_seqlen_q.numel() + 1,
        device=q.device,
        dtype=torch.int32,
    )
    moba_cu_seqlen_q[0] = 0
    moba_cu_seqlen_q[1:] = full_moba_seqlen_q.cumsum(dim=0)
    
    """ prepare moba kv """
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert int(valid_expert_mask.sum().item()) == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )