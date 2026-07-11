# Copyright 2026 vLLM-HUST.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Segment-reuse terminal replay mask contracts for Ascend backends.

This module owns the operator/runtime contract for body-window terminal replay.
It is intentionally a pure Python/PyTorch layer first: vLLM-Ascend can call it
as a fail-closed contract checker while a lower-level Triton/C++ primitive is
introduced behind the same semantics.
"""

from __future__ import annotations

import torch

ASCEND_FIA_SPLITFUSE_BODY_WINDOW_PROOF = (
    "ascend_fia_splitfuse_sparse_mode_3_body_window_v1"
)
ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF = (
    "ascend_terminal_replay_body_window_output_differential_v1"
)


def _mask_values(dtype: torch.dtype) -> tuple[object, object]:
    if dtype == torch.bool:
        return True, False
    if dtype.is_floating_point:
        mask_value = float("-inf") if dtype == torch.float16 else torch.finfo(dtype).min
        return mask_value, 0
    return 1, 0


def generate_segment_reuse_body_isolation_mask(
    max_seq_len: int,
    envelope_token_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a causal mask where body rows cannot attend envelope columns.

    Zero/false means visible. A dtype-specific nonzero value means hidden.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if envelope_token_count <= 0 or envelope_token_count >= max_seq_len:
        raise ValueError("envelope_token_count must split envelope and body")

    mask_value, visible_value = _mask_values(dtype)
    mask = torch.full((max_seq_len, max_seq_len), visible_value, dtype=dtype)
    causal_hidden = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).triu_(1)
    mask.masked_fill_(causal_hidden, mask_value)
    mask[envelope_token_count:, :envelope_token_count] = mask_value
    return mask


def generate_segment_reuse_splitfuse_body_isolation_mask(
    envelope_token_count: int,
    dtype: torch.dtype = torch.int8,
    mask_len: int = 2048,
) -> torch.Tensor:
    """Generate a body-isolation mask with Ascend split-fuse FIA shape."""

    if mask_len <= 0:
        raise ValueError("mask_len must be positive")
    if envelope_token_count <= 0 or envelope_token_count >= mask_len:
        raise ValueError("envelope_token_count must fit within split-fuse mask")

    return generate_segment_reuse_body_isolation_mask(
        max_seq_len=mask_len,
        envelope_token_count=envelope_token_count,
        dtype=dtype,
    )


def generate_segment_reuse_terminal_replay_mask(
    max_seq_len: int,
    envelope_token_count: int,
    query_start_token: int,
    query_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a query-offset mask for cached-body terminal replay.

    The first query rows are rewritten as absolute rows at
    ``query_start_token + row`` while preserving body isolation.
    """

    if query_start_token < 0:
        raise ValueError("query_start_token must be non-negative")
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    if query_start_token + query_tokens > max_seq_len:
        raise ValueError("terminal replay query exceeds max_seq_len")

    mask = generate_segment_reuse_body_isolation_mask(
        max_seq_len=max_seq_len,
        envelope_token_count=envelope_token_count,
        dtype=dtype,
    )
    mask_value, visible_value = _mask_values(dtype)

    cols = torch.arange(max_seq_len)
    for row in range(query_tokens):
        absolute_pos = query_start_token + row
        for mask_row in {row, absolute_pos}:
            mask[mask_row].fill_(visible_value)
            mask[mask_row, :envelope_token_count] = mask_value
            mask[mask_row, cols > absolute_pos] = mask_value
    return mask


def generate_segment_reuse_splitfuse_terminal_replay_mask(
    envelope_token_count: int,
    query_start_token: int,
    query_tokens: int,
    dtype: torch.dtype = torch.int8,
    mask_len: int = 2048,
) -> torch.Tensor:
    """Generate an Ascend split-fuse terminal replay mask.

    Ascend FIA split-fuse sparse_mode=3 requires a 2048x2048 2-D mask even
    when the logical sequence is longer. Terminal replay query rows are encoded
    in the first ``query_tokens`` rows. Columns address the split-fuse body
    window, so all non-envelope columns in the window are visible.
    """

    if query_start_token < envelope_token_count:
        raise ValueError("query_start_token must be in the body window")
    if query_tokens <= 0:
        raise ValueError("query_tokens must be positive")
    if query_tokens > mask_len:
        raise ValueError("query_tokens must fit within split-fuse mask")

    mask = generate_segment_reuse_splitfuse_body_isolation_mask(
        envelope_token_count=envelope_token_count,
        dtype=dtype,
        mask_len=mask_len,
    )
    mask_value, visible_value = _mask_values(dtype)
    for row in range(query_tokens):
        mask[row].fill_(visible_value)
        mask[row, :envelope_token_count] = mask_value
    return mask


def _segment_reuse_masked(mask_row: torch.Tensor) -> torch.Tensor:
    if mask_row.dtype == torch.bool:
        return mask_row
    return mask_row != 0


def _segment_reuse_row_matches_body_window(
    attn_mask: torch.Tensor,
    *,
    row: int,
    absolute_pos: int,
    envelope_token_count: int,
    cols: int,
) -> bool:
    if row < 0 or row >= int(attn_mask.shape[0]):
        return False
    row_mask = _segment_reuse_masked(attn_mask[row, :cols])
    if not row_mask[:envelope_token_count].all().item():
        return False
    if absolute_pos + 1 < cols:
        if not row_mask[absolute_pos + 1 : cols].all().item():
            return False
    body_visible_start = envelope_token_count
    body_visible_end = min(absolute_pos + 1, cols)
    if body_visible_start < body_visible_end:
        if row_mask[body_visible_start:body_visible_end].any().item():
            return False
    return True


def prove_segment_reuse_terminal_replay_mask_semantics(
    attn_mask: torch.Tensor | None,
    *,
    envelope_token_count: int,
    terminal_query_start: int,
    terminal_query_tokens: int,
    sparse_mode: int,
    expected_mask_len: int | None = None,
) -> tuple[bool, str]:
    """Validate the body-window mask contract consumed by Ascend FIA."""

    if terminal_query_tokens <= 0:
        return True, "terminal_replay_not_required"
    if sparse_mode != 3:
        return False, "terminal_replay_sparse_mode_unsupported"
    if attn_mask is None:
        return False, "terminal_replay_mask_missing"
    if attn_mask.dim() != 2:
        return False, "terminal_replay_mask_rank_unsupported"

    rows = int(attn_mask.shape[0])
    cols = int(attn_mask.shape[1])
    if rows != cols:
        return False, "terminal_replay_mask_shape_unsupported"
    if expected_mask_len is not None and rows != expected_mask_len:
        return False, "terminal_replay_mask_shape_unsupported"
    if envelope_token_count <= 0 or envelope_token_count >= cols:
        return False, "terminal_replay_envelope_tokens_invalid"
    if terminal_query_start < envelope_token_count:
        return False, "terminal_replay_query_start_before_body"
    splitfuse_window = (
        rows == 2048
        and cols == 2048
        and terminal_query_start + terminal_query_tokens > cols
    )
    if terminal_query_start + terminal_query_tokens > cols and not splitfuse_window:
        return False, "terminal_replay_query_window_invalid"

    for query_row in range(terminal_query_tokens):
        absolute_pos = min(terminal_query_start + query_row, cols - 1)
        if not _segment_reuse_row_matches_body_window(
            attn_mask,
            row=query_row,
            absolute_pos=absolute_pos,
            envelope_token_count=envelope_token_count,
            cols=cols,
        ):
            return False, "terminal_replay_query_row_contract_failed"
        if terminal_query_start + query_row < rows:
            if not _segment_reuse_row_matches_body_window(
                attn_mask,
                row=terminal_query_start + query_row,
                absolute_pos=absolute_pos,
                envelope_token_count=envelope_token_count,
                cols=cols,
            ):
                return False, "terminal_replay_absolute_row_contract_failed"

    return True, ASCEND_FIA_SPLITFUSE_BODY_WINDOW_PROOF


def prove_segment_reuse_terminal_replay_output_semantics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend_output: torch.Tensor,
    *,
    envelope_token_count: int,
    terminal_query_start: int,
    terminal_query_tokens: int,
    context_tokens: int | None = None,
    scale: float | None = None,
    reference_atol: float = 2e-2,
    num_query_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_size: int | None = None,
) -> dict[str, object]:
    """Differentially prove terminal replay output matches body-window semantics.

    This is the runtime/output counterpart to the mask contract checker above.
    It compares the backend output against a body-window reference and also
    verifies that an all-context causal reference is distinguishable.
    """

    try:
        query_tnd = _as_tnd(query, num_query_heads, head_size, "query")
        key_tnd = _as_tnd(key, num_kv_heads, head_size, "key")
        value_tnd = _as_tnd(value, num_kv_heads, head_size, "value")
        backend_tnd = _as_tnd(
            backend_output,
            num_query_heads,
            head_size,
            "backend_output",
        )
        if context_tokens is None:
            context_tokens = int(key_tnd.shape[0])
        _validate_terminal_replay_tensors(
            query_tnd,
            key_tnd,
            value_tnd,
            backend_tnd,
            envelope_token_count=envelope_token_count,
            terminal_query_start=terminal_query_start,
            terminal_query_tokens=terminal_query_tokens,
            context_tokens=context_tokens,
        )
        key_tnd = _expand_kv_heads(key_tnd, int(query_tnd.shape[1]))
        value_tnd = _expand_kv_heads(value_tnd, int(query_tnd.shape[1]))
        query_tnd = query_tnd[:terminal_query_tokens]
        backend_tnd = backend_tnd[:terminal_query_tokens]
        key_tnd = key_tnd[:context_tokens]
        value_tnd = value_tnd[:context_tokens]
        reference = _body_window_reference_attention(
            query_tnd,
            key_tnd,
            value_tnd,
            envelope_token_count=envelope_token_count,
            terminal_query_start=terminal_query_start,
            scale=scale,
        )
        all_context = _causal_all_context_attention(
            query_tnd,
            key_tnd,
            value_tnd,
            terminal_query_start=terminal_query_start,
            scale=scale,
        )
        reference_delta = (backend_tnd.float() - reference.float()).abs()
        all_context_delta = (backend_tnd.float() - all_context.float()).abs()
        hidden_max = float(reference_delta.max().item())
        hidden_mean = float(reference_delta.mean().item())
        all_context_max = float(all_context_delta.max().item())
        all_context_mean = float(all_context_delta.mean().item())
        semantic_proven = hidden_max <= reference_atol and all_context_max > 0
        return {
            "path_executed": True,
            "semantic_proven": bool(semantic_proven),
            "semantic_reason": (
                ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF
                if semantic_proven
                else "terminal_replay_output_reference_mismatch"
            ),
            "proof_reason": (
                ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF
                if semantic_proven
                else "terminal_replay_output_reference_mismatch"
            ),
            "hidden_max_abs_diff": hidden_max,
            "hidden_mean_abs_diff": hidden_mean,
            "logits_max_abs_diff": None,
            "runtime_logits_max_abs_diff": None,
            "all_context_max_abs_diff": all_context_max,
            "all_context_mean_abs_diff": all_context_mean,
            "reference_atol": float(reference_atol),
            "output_shape": list(backend_tnd.shape),
            "query_shape": list(query_tnd.shape),
            "key_shape": list(key_tnd.shape),
            "value_shape": list(value_tnd.shape),
            "envelope_tokens": int(envelope_token_count),
            "terminal_query_start": int(terminal_query_start),
            "terminal_query_tokens": int(terminal_query_tokens),
            "context_tokens": int(context_tokens),
        }
    except Exception as exc:
        return {
            "path_executed": False,
            "semantic_proven": False,
            "semantic_reason": "terminal_replay_output_differential_failed",
            "proof_reason": "terminal_replay_output_differential_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def materialize_segment_reuse_terminal_replay_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    block_table: torch.Tensor | None,
    req_idx: int,
    context_tokens: int,
    block_size: int,
    terminal_query_tokens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Materialize the logical terminal-replay proof tensors from paged KV.

    The caller supplies raw runtime tensors and metadata. This Triton-owned
    contract layer performs the paged-block interpretation so serving glue does
    not duplicate KV layout semantics.
    """

    if block_table is None:
        raise ValueError("terminal replay block_table missing")
    if req_idx < 0 or req_idx >= int(block_table.shape[0]):
        raise ValueError("terminal replay boundary req_idx outside block_table")
    if context_tokens <= 0:
        raise ValueError("terminal replay context_tokens invalid")
    if block_size <= 0:
        raise ValueError("terminal replay block_size invalid")
    if terminal_query_tokens <= 0:
        raise ValueError("terminal replay terminal_query_tokens invalid")

    query_tnd = _as_tnd(query, num_query_heads, head_size, "query")
    if int(query_tnd.shape[0]) < int(terminal_query_tokens):
        raise ValueError("query has fewer rows than terminal_query_tokens")

    blocks_needed = (int(context_tokens) + int(block_size) - 1) // int(block_size)
    if blocks_needed > int(block_table.shape[1]):
        raise ValueError("terminal replay block_table lacks context blocks")
    block_ids = block_table[int(req_idx), :blocks_needed].to(
        device=key.device,
        dtype=torch.long,
    )
    key_blocks = torch.index_select(key, 0, block_ids)
    value_blocks = torch.index_select(value, 0, block_ids)
    key_tnd = key_blocks.reshape(-1, int(num_kv_heads), int(head_size))[
        : int(context_tokens)
    ].clone()
    value_tnd = value_blocks.reshape(-1, int(num_kv_heads), int(head_size))[
        : int(context_tokens)
    ].clone()
    query_tnd = query_tnd[: int(terminal_query_tokens)].clone()
    return (
        query_tnd,
        key_tnd,
        value_tnd,
        {
            "logical_kv_source": "triton-ascend-paged-cache-block-table",
            "boundary_req_idx": int(req_idx),
            "block_size": int(block_size),
            "context_tokens": int(context_tokens),
            "query_heads": int(num_query_heads),
            "kv_heads": int(num_kv_heads),
            "head_size": int(head_size),
        },
    )


def _as_tnd(
    tensor: torch.Tensor,
    heads: int | None,
    head_size: int | None,
    name: str,
) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 2 and heads is not None and head_size is not None:
        if tensor.shape[1] != heads * head_size:
            raise ValueError(f"{name} hidden width does not match heads*head_size")
        return tensor.view(tensor.shape[0], heads, head_size)
    raise ValueError(f"{name} must be [tokens, heads, head_dim] or flattenable TND")


def _expand_kv_heads(tensor: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = int(tensor.shape[1])
    if num_kv_heads == num_query_heads:
        return tensor
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be a multiple of kv heads")
    return tensor.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


def _validate_terminal_replay_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend_output: torch.Tensor,
    *,
    envelope_token_count: int,
    terminal_query_start: int,
    terminal_query_tokens: int,
    context_tokens: int,
) -> None:
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError("query/key/value must be TND tensors")
    if key.shape != value.shape:
        raise ValueError("key/value shape mismatch")
    if backend_output.dim() != 3:
        raise ValueError("backend_output must be a TND tensor")
    if terminal_query_tokens <= 0:
        raise ValueError("terminal_query_tokens must be positive")
    if envelope_token_count <= 0:
        raise ValueError("envelope_token_count must be positive")
    if terminal_query_start < envelope_token_count:
        raise ValueError("terminal_query_start must be inside the body window")
    if terminal_query_start + terminal_query_tokens > context_tokens:
        raise ValueError("terminal replay query window exceeds context")
    if context_tokens > key.shape[0]:
        raise ValueError("context_tokens exceeds key/value length")
    if query.shape[0] < terminal_query_tokens:
        raise ValueError("query has fewer rows than terminal_query_tokens")
    if backend_output.shape[0] < terminal_query_tokens:
        raise ValueError("backend output has fewer rows than terminal_query_tokens")
    if backend_output.shape[1:] != query.shape[1:]:
        raise ValueError("backend output head layout mismatch")


def _body_window_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    envelope_token_count: int,
    terminal_query_start: int,
    scale: float | None,
) -> torch.Tensor:
    cols = torch.arange(key.shape[0], device=query.device)
    visibility = []
    for row in range(query.shape[0]):
        absolute_pos = terminal_query_start + row
        visibility.append((cols >= envelope_token_count) & (cols <= absolute_pos))
    return _masked_attention(query, key, value, torch.stack(visibility), scale=scale)


def _causal_all_context_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    terminal_query_start: int,
    scale: float | None,
) -> torch.Tensor:
    cols = torch.arange(key.shape[0], device=query.device)
    visibility = []
    for row in range(query.shape[0]):
        visibility.append(cols <= terminal_query_start + row)
    return _masked_attention(query, key, value, torch.stack(visibility), scale=scale)


def _masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    visibility: torch.Tensor,
    *,
    scale: float | None,
) -> torch.Tensor:
    if visibility.shape != (query.shape[0], key.shape[0]):
        raise ValueError("visibility shape mismatch")
    if scale is None:
        scale = query.shape[-1] ** -0.5
    rows = []
    for row in range(query.shape[0]):
        scores = (key.float() * query[row].float().unsqueeze(0)).sum(dim=-1)
        scores = scores.transpose(0, 1) * float(scale)
        scores = scores.masked_fill(~visibility[row].unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = (weights.transpose(0, 1).unsqueeze(-1) * value.float()).sum(dim=0)
        rows.append(output)
    return torch.stack(rows, dim=0).to(query.dtype)
