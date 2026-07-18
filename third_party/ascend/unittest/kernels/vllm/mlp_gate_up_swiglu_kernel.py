"""Dense gate/up projection with an in-kernel SwiGLU epilogue.

This carrier intentionally stops at the activation workspace.  Unlike the
full-MLP private-lifetime prototype, output partitioning never duplicates the
gate/up projection.  The following down projection remains a vendor GEMM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gate_up_swiglu_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    activation_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    stride_hidden_m: tl.constexpr,
    stride_hidden_h: tl.constexpr,
    stride_weight_row: tl.constexpr,
    stride_weight_h: tl.constexpr,
    stride_activation_m: tl.constexpr,
    stride_activation_i: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Write only the post-SwiGLU activation, never the packed gate/up."""

    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hidden = tl.load(
            hidden_ptr
            + offs_m[:, None] * stride_hidden_m
            + offs_h[None, :] * stride_hidden_h,
            mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )
        gate_weight = tl.load(
            gate_up_weight_ptr
            + offs_i[None, :] * stride_weight_row
            + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        up_weight = tl.load(
            gate_up_weight_ptr
            + (I + offs_i[None, :]) * stride_weight_row
            + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        gate_acc = tl.dot(hidden, gate_weight, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(hidden, up_weight, up_acc, out_dtype=tl.float32)

    activation = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
    tl.store(
        activation_ptr
        + offs_m[:, None] * stride_activation_m
        + offs_i[None, :] * stride_activation_i,
        activation.to(activation_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < I),
    )


def gate_up_swiglu(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    *,
    block_m: int = 16,
    block_h: int = 128,
    block_i: int = 128,
) -> torch.Tensor:
    """Run dense ``linear -> SwiGLU`` with no packed gate/up writeback."""

    assert hidden.ndim == 2
    assert gate_up_weight.ndim == 2
    assert hidden.is_contiguous()
    assert gate_up_weight.is_contiguous()
    assert hidden.dtype in (torch.float16, torch.bfloat16)
    assert gate_up_weight.dtype == hidden.dtype

    m, h = hidden.shape
    gate_up_rows, weight_h = gate_up_weight.shape
    assert gate_up_rows % 2 == 0
    assert weight_h == h
    intermediate = gate_up_rows // 2

    activation = torch.empty(
        (m, intermediate), device=hidden.device, dtype=hidden.dtype
    )
    grid = (triton.cdiv(m, block_m), triton.cdiv(intermediate, block_i))
    gate_up_swiglu_kernel[grid](
        hidden,
        gate_up_weight,
        activation,
        m,
        h,
        intermediate,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        activation.stride(0),
        activation.stride(1),
        block_m,
        block_h,
        block_i,
    )
    return activation
