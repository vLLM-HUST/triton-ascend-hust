"""Matched dense BF16 gate/up materialization pair for Ascend Triton.

``Tmat`` and ``Tfused`` deliberately share one :class:`ScheduleIdentity`.
Only the producer epilogue differs: Tmat writes packed gate/up and invokes the
native SwiGLU operator, while Tfused applies SwiGLU before the producer store.
Both arms use the same down projection and caller-owned timing boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class ScheduleIdentity:
    block_m: int
    block_h: int
    block_i: int

    def __post_init__(self) -> None:
        if self.block_m not in (8, 16, 32):
            raise ValueError("block_m is outside the frozen candidate space")
        if self.block_h not in (64, 128, 256):
            raise ValueError("block_h is outside the frozen candidate space")
        if self.block_i not in (64, 128, 256):
            raise ValueError("block_i is outside the frozen candidate space")

    @property
    def candidate_id(self) -> str:
        return f"bm{self.block_m}-bh{self.block_h}-bi{self.block_i}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScheduleIdentity":
        allowed = {"block_m", "block_h", "block_i", "candidate_id", "selected_from_arm"}
        if set(value) != allowed or value.get("selected_from_arm") != "Tmat":
            raise ValueError("schedule identity must be selected exclusively from Tmat")
        result = cls(int(value["block_m"]), int(value["block_h"]), int(value["block_i"]))
        if value.get("candidate_id") != result.candidate_id:
            raise ValueError("candidate_id does not match the frozen schedule")
        return result


@triton.jit
def _gate_up_materializing_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    packed_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    stride_hidden_m: tl.constexpr,
    stride_hidden_h: tl.constexpr,
    stride_weight_row: tl.constexpr,
    stride_weight_h: tl.constexpr,
    stride_packed_m: tl.constexpr,
    stride_packed_i: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_h[None, :] * stride_hidden_h,
            mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )
        gate_weight = tl.load(
            gate_up_weight_ptr + offs_i[None, :] * stride_weight_row + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        up_weight = tl.load(
            gate_up_weight_ptr + (I + offs_i[None, :]) * stride_weight_row + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        gate_acc = tl.dot(hidden, gate_weight, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(hidden, up_weight, up_acc, out_dtype=tl.float32)

    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)
    tl.store(
        packed_ptr + offs_m[:, None] * stride_packed_m + offs_i[None, :] * stride_packed_i,
        gate_acc.to(packed_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        packed_ptr + offs_m[:, None] * stride_packed_m + (I + offs_i[None, :]) * stride_packed_i,
        up_acc.to(packed_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _gate_up_fused_kernel(
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
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hidden = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hidden_m + offs_h[None, :] * stride_hidden_h,
            mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )
        gate_weight = tl.load(
            gate_up_weight_ptr + offs_i[None, :] * stride_weight_row + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        up_weight = tl.load(
            gate_up_weight_ptr + (I + offs_i[None, :]) * stride_weight_row + offs_h[:, None] * stride_weight_h,
            mask=(offs_i[None, :] < I) & (offs_h[:, None] < H),
            other=0.0,
        )
        gate_acc = tl.dot(hidden, gate_weight, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(hidden, up_weight, up_acc, out_dtype=tl.float32)

    activation = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
    tl.store(
        activation_ptr + offs_m[:, None] * stride_activation_m + offs_i[None, :] * stride_activation_i,
        activation.to(activation_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < I),
    )


def _validate_inputs(hidden: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> tuple[int, int, int]:
    if hidden.ndim != 2 or gate_up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("matched MLP inputs must be rank two")
    if not hidden.is_contiguous() or not gate_up_weight.is_contiguous() or not down_weight.is_contiguous():
        raise ValueError("matched MLP inputs must be contiguous")
    if hidden.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("matched MLP supports FP16/BF16 only")
    if gate_up_weight.dtype != hidden.dtype or down_weight.dtype != hidden.dtype:
        raise ValueError("matched MLP dtypes differ")
    m, hidden_size = hidden.shape
    packed_rows, weight_hidden = gate_up_weight.shape
    output, intermediate = down_weight.shape
    if packed_rows != 2 * intermediate or weight_hidden != hidden_size or output != hidden_size:
        raise ValueError("matched MLP shape contract differs from dense Qwen MLP")
    return m, hidden_size, intermediate


def _launch_materializing(hidden: torch.Tensor, weight: torch.Tensor, schedule: ScheduleIdentity, intermediate: int) -> torch.Tensor:
    packed = torch.empty((hidden.shape[0], 2 * intermediate), device=hidden.device, dtype=hidden.dtype)
    grid = (triton.cdiv(hidden.shape[0], schedule.block_m), triton.cdiv(intermediate, schedule.block_i))
    _gate_up_materializing_kernel[grid](
        hidden, weight, packed, hidden.shape[0], hidden.shape[1], intermediate,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1),
        packed.stride(0), packed.stride(1), schedule.block_m, schedule.block_h, schedule.block_i,
    )
    return packed


def _launch_fused(hidden: torch.Tensor, weight: torch.Tensor, schedule: ScheduleIdentity, intermediate: int) -> torch.Tensor:
    activation = torch.empty((hidden.shape[0], intermediate), device=hidden.device, dtype=hidden.dtype)
    grid = (triton.cdiv(hidden.shape[0], schedule.block_m), triton.cdiv(intermediate, schedule.block_i))
    _gate_up_fused_kernel[grid](
        hidden, weight, activation, hidden.shape[0], hidden.shape[1], intermediate,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1),
        activation.stride(0), activation.stride(1), schedule.block_m, schedule.block_h, schedule.block_i,
    )
    return activation


def tmat(hidden: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor, schedule: ScheduleIdentity) -> torch.Tensor:
    """Materialize packed gate/up under the Tmat-selected schedule."""
    _, _, intermediate = _validate_inputs(hidden, gate_up_weight, down_weight)
    packed = _launch_materializing(hidden, gate_up_weight, schedule, intermediate)
    import torch_npu

    activation = torch_npu.npu_swiglu(packed)
    return torch.nn.functional.linear(activation, down_weight)

def tfused(hidden: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor, schedule: ScheduleIdentity) -> torch.Tensor:
    """Eliminate only the packed gate/up boundary using the same identity."""
    _, _, intermediate = _validate_inputs(hidden, gate_up_weight, down_weight)
    activation = _launch_fused(hidden, gate_up_weight, schedule, intermediate)
    return torch.nn.functional.linear(activation, down_weight)
