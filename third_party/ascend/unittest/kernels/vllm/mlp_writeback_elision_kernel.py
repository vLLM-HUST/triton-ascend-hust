import json

import torch
import triton
import triton.language as tl


@triton.jit
def mlp_writeback_elision_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    down_weight_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    stride_hidden_m: tl.constexpr,
    stride_hidden_h: tl.constexpr,
    stride_gate_up_row: tl.constexpr,
    stride_gate_up_h: tl.constexpr,
    stride_down_o: tl.constexpr,
    stride_down_i: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_o: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Tiny dense MLP carrier prototype for fixed/small shapes.

    This kernel intentionally keeps the SwiGLU activation inside the kernel.
    It never stores the gate/up or activation tensors as graph-visible/global
    outputs.  It is a feasibility slice, not a tuned Qwen2-size implementation.
    """

    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)

    acc_out = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
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
            gate_w = tl.load(
                gate_up_weight_ptr
                + offs_i[None, :] * stride_gate_up_row
                + offs_h[:, None] * stride_gate_up_h,
                mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
                other=0.0,
            )
            up_w = tl.load(
                gate_up_weight_ptr
                + (offs_i[None, :] + I) * stride_gate_up_row
                + offs_h[:, None] * stride_gate_up_h,
                mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
                other=0.0,
            )
            gate_acc = tl.dot(hidden, gate_w, gate_acc, out_dtype=tl.float32)
            up_acc = tl.dot(hidden, up_w, up_acc, out_dtype=tl.float32)

        activation = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
        down_w = tl.load(
            down_weight_ptr
            + offs_o[:, None] * stride_down_o
            + offs_i[None, :] * stride_down_i,
            mask=(offs_o[:, None] < O) & (offs_i[None, :] < I),
            other=0.0,
        )
        activation = activation.to(down_w.dtype)
        acc_out = tl.dot(activation, tl.trans(down_w), acc_out, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_o[None, :] * stride_out_o,
        acc_out.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_o[None, :] < O),
    )


@triton.jit
def mlp_writeback_elision_group2_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    down_weight_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    stride_hidden_m: tl.constexpr,
    stride_hidden_h: tl.constexpr,
    stride_gate_up_row: tl.constexpr,
    stride_gate_up_h: tl.constexpr,
    stride_down_o: tl.constexpr,
    stride_down_i: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_o: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Compute two output tiles per program while sharing gate/up work."""

    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o0 = (pid_o * 2) * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_o1 = offs_o0 + BLOCK_O
    acc_out0 = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)
    acc_out1 = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
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
            gate_w = tl.load(
                gate_up_weight_ptr
                + offs_i[None, :] * stride_gate_up_row
                + offs_h[:, None] * stride_gate_up_h,
                mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
                other=0.0,
            )
            up_w = tl.load(
                gate_up_weight_ptr
                + (offs_i[None, :] + I) * stride_gate_up_row
                + offs_h[:, None] * stride_gate_up_h,
                mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
                other=0.0,
            )
            gate_acc = tl.dot(hidden, gate_w, gate_acc, out_dtype=tl.float32)
            up_acc = tl.dot(hidden, up_w, up_acc, out_dtype=tl.float32)

        activation = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
        down_w0 = tl.load(
            down_weight_ptr
            + offs_o0[:, None] * stride_down_o
            + offs_i[None, :] * stride_down_i,
            mask=(offs_o0[:, None] < O) & (offs_i[None, :] < I),
            other=0.0,
        )
        down_w1 = tl.load(
            down_weight_ptr
            + offs_o1[:, None] * stride_down_o
            + offs_i[None, :] * stride_down_i,
            mask=(offs_o1[:, None] < O) & (offs_i[None, :] < I),
            other=0.0,
        )
        activation = activation.to(down_w0.dtype)
        acc_out0 = tl.dot(activation, tl.trans(down_w0), acc_out0, out_dtype=tl.float32)
        acc_out1 = tl.dot(activation, tl.trans(down_w1), acc_out1, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_o0[None, :] * stride_out_o,
        acc_out0.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_o0[None, :] < O),
    )
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_o1[None, :] * stride_out_o,
        acc_out1.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_o1[None, :] < O),
    )


def mlp_writeback_elision(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    block_m: int = 8,
    block_o: int = 16,
    block_h: int = 32,
    block_i: int = 32,
) -> torch.Tensor:
    """Launch the tiny no-bias dense MLP carrier.

    Shape contract:
      hidden: [M, H]
      gate_up_weight: [2 * I, H], with gate rows first and up rows second
      down_weight: [O, I]
    """

    assert hidden.ndim == 2
    assert gate_up_weight.ndim == 2
    assert down_weight.ndim == 2
    assert hidden.is_contiguous()
    assert gate_up_weight.is_contiguous()
    assert down_weight.is_contiguous()
    assert hidden.dtype in (torch.float16, torch.bfloat16)
    assert gate_up_weight.dtype == hidden.dtype
    assert down_weight.dtype == hidden.dtype

    m, h = hidden.shape
    gate_up_rows, gate_up_h = gate_up_weight.shape
    o, i = down_weight.shape
    assert gate_up_rows == 2 * i
    assert gate_up_h == h

    out = torch.empty((m, o), device=hidden.device, dtype=hidden.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(o, block_o))
    mlp_writeback_elision_kernel[grid](
        hidden,
        gate_up_weight,
        down_weight,
        out,
        m,
        h,
        i,
        o,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        out.stride(0),
        out.stride(1),
        block_m,
        block_o,
        block_h,
        block_i,
    )
    return out


def mlp_writeback_elision_group2(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    block_m: int = 8,
    block_o: int = 128,
    block_h: int = 128,
    block_i: int = 128,
) -> torch.Tensor:
    """Launch the two-output-tile reuse experiment."""

    assert hidden.ndim == 2
    assert gate_up_weight.ndim == 2
    assert down_weight.ndim == 2
    assert hidden.is_contiguous()
    assert gate_up_weight.is_contiguous()
    assert down_weight.is_contiguous()
    assert hidden.dtype in (torch.float16, torch.bfloat16)
    assert gate_up_weight.dtype == hidden.dtype
    assert down_weight.dtype == hidden.dtype

    m, h = hidden.shape
    gate_up_rows, gate_up_h = gate_up_weight.shape
    o, i = down_weight.shape
    assert gate_up_rows == 2 * i
    assert gate_up_h == h

    out = torch.empty((m, o), device=hidden.device, dtype=hidden.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(o, 2 * block_o))
    mlp_writeback_elision_group2_kernel[grid](
        hidden,
        gate_up_weight,
        down_weight,
        out,
        m,
        h,
        i,
        o,
        hidden.stride(0),
        hidden.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        out.stride(0),
        out.stride(1),
        block_m,
        block_o,
        block_h,
        block_i,
    )
    return out


def reference_mlp(hidden: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = hidden @ gate_up_weight.T
    gate, up = gate_up.chunk(2, dim=-1)
    activation = torch.nn.functional.silu(gate) * up
    return activation @ down_weight.T


def smoke_npu6() -> None:
    """Manual correctness probe. Run only with NPU6 exposed."""

    torch.manual_seed(0)
    base_hidden = torch.randn((4, 32))
    base_gate_up_weight = torch.randn((64, 32))
    base_down_weight = torch.randn((16, 32))

    cases = (
        (torch.bfloat16, 5e-2, 1.0, 1e-2),
        (torch.float16, 2e-2, 1e-1, 5e-3),
    )
    results = []
    for dtype, rtol, atol, max_rel_l2 in cases:
        hidden_cpu = base_hidden.to(dtype)
        gate_up_weight_cpu = base_gate_up_weight.to(dtype)
        down_weight_cpu = base_down_weight.to(dtype)
        hidden = hidden_cpu.npu()
        gate_up_weight = gate_up_weight_cpu.npu()
        down_weight = down_weight_cpu.npu()

        got = mlp_writeback_elision(hidden, gate_up_weight, down_weight)
        torch.npu.synchronize()
        got_cpu = got.cpu().float()
        ref_cpu = reference_mlp(
            hidden_cpu.float(), gate_up_weight_cpu.float(), down_weight_cpu.float()
        )
        diff = got_cpu - ref_cpu
        rel_l2 = (torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref_cpu)).item()
        metrics = {
            "dtype": str(dtype),
            "max_abs": diff.abs().max().item(),
            "rmse": diff.square().mean().sqrt().item(),
            "rel_l2": rel_l2,
        }
        results.append(metrics)
        torch.testing.assert_close(got_cpu, ref_cpu, rtol=rtol, atol=atol)
        assert rel_l2 <= max_rel_l2, metrics

    print(json.dumps({"status": "ok", "cases": results}, sort_keys=True))


if __name__ == "__main__":
    smoke_npu6()
