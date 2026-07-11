import pytest
import torch

from triton.extension.segment_reuse_terminal_replay import (
    ASCEND_FIA_SPLITFUSE_BODY_WINDOW_PROOF,
    ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF,
    generate_segment_reuse_body_isolation_mask,
    generate_segment_reuse_splitfuse_body_isolation_mask,
    generate_segment_reuse_splitfuse_terminal_replay_mask,
    generate_segment_reuse_terminal_replay_mask,
    materialize_segment_reuse_terminal_replay_tensors,
    prove_segment_reuse_terminal_replay_output_semantics,
    prove_segment_reuse_terminal_replay_mask_semantics,
)


def _masked(row: torch.Tensor) -> torch.Tensor:
    if row.dtype == torch.bool:
        return row
    return row != 0


def test_body_isolation_mask_hides_envelope_from_body_rows():
    mask = generate_segment_reuse_body_isolation_mask(
        max_seq_len=16,
        envelope_token_count=4,
        dtype=torch.int8,
    )

    body_row = _masked(mask[7])
    assert body_row[:4].all()
    assert not body_row[4:8].any()
    assert body_row[8:].all()

    envelope_row = _masked(mask[2])
    assert not envelope_row[:3].any()
    assert envelope_row[3:].all()


def test_splitfuse_body_isolation_mask_preserves_2048_shape():
    mask = generate_segment_reuse_splitfuse_body_isolation_mask(
        envelope_token_count=128,
        dtype=torch.int8,
    )

    assert tuple(mask.shape) == (2048, 2048)
    body_row = _masked(mask[130])
    assert body_row[:128].all()
    assert not body_row[128:131].any()
    assert body_row[131:].all()


def test_terminal_replay_mask_rewrites_query_offset_rows():
    mask = generate_segment_reuse_terminal_replay_mask(
        max_seq_len=2048,
        envelope_token_count=256,
        query_start_token=384,
        query_tokens=3,
        dtype=torch.int8,
    )

    ok, reason = prove_segment_reuse_terminal_replay_mask_semantics(
        mask,
        envelope_token_count=256,
        terminal_query_start=384,
        terminal_query_tokens=3,
        sparse_mode=3,
    )
    assert ok, reason
    assert reason == ASCEND_FIA_SPLITFUSE_BODY_WINDOW_PROOF

    query_row = _masked(mask[0])
    assert query_row[:256].all()
    assert not query_row[256:385].any()
    assert query_row[385:].all()

    absolute_row = _masked(mask[384])
    assert absolute_row[:256].all()
    assert not absolute_row[256:385].any()
    assert absolute_row[385:].all()


def test_splitfuse_terminal_replay_mask_keeps_ascend_required_shape():
    mask = generate_segment_reuse_splitfuse_terminal_replay_mask(
        envelope_token_count=256,
        query_start_token=2304,
        query_tokens=8,
        dtype=torch.int8,
    )

    assert tuple(mask.shape) == (2048, 2048)
    query_row = _masked(mask[0])
    assert query_row[:256].all()
    assert not query_row[256:].any()

    ok, reason = prove_segment_reuse_terminal_replay_mask_semantics(
        mask,
        envelope_token_count=256,
        terminal_query_start=2304,
        terminal_query_tokens=8,
        sparse_mode=3,
        expected_mask_len=2048,
    )
    assert ok, reason
    assert reason == ASCEND_FIA_SPLITFUSE_BODY_WINDOW_PROOF


@pytest.mark.parametrize(
    ("mask", "sparse_mode", "reason"),
    [
        (None, 3, "terminal_replay_mask_missing"),
        (torch.zeros((2048, 2048), dtype=torch.int8), 4, "terminal_replay_sparse_mode_unsupported"),
        (torch.zeros((32, 32), dtype=torch.int8), 3, "terminal_replay_mask_shape_unsupported"),
    ],
)
def test_terminal_replay_proof_fail_closed_for_invalid_operator_contract(
    mask,
    sparse_mode,
    reason,
):
    ok, got_reason = prove_segment_reuse_terminal_replay_mask_semantics(
        mask,
        envelope_token_count=8,
        terminal_query_start=16,
        terminal_query_tokens=1,
        sparse_mode=sparse_mode,
        expected_mask_len=2048,
    )

    assert not ok
    assert got_reason == reason


def test_terminal_replay_proof_fail_closed_for_invalid_body_window():
    mask = generate_segment_reuse_terminal_replay_mask(
        max_seq_len=2048,
        envelope_token_count=256,
        query_start_token=384,
        query_tokens=1,
        dtype=torch.int8,
    )

    ok, reason = prove_segment_reuse_terminal_replay_mask_semantics(
        mask,
        envelope_token_count=256,
        terminal_query_start=128,
        terminal_query_tokens=1,
        sparse_mode=3,
    )

    assert not ok
    assert reason == "terminal_replay_query_start_before_body"


def test_terminal_replay_proof_detects_all_context_row():
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1)

    ok, reason = prove_segment_reuse_terminal_replay_mask_semantics(
        mask,
        envelope_token_count=256,
        terminal_query_start=384,
        terminal_query_tokens=1,
        sparse_mode=3,
    )

    assert not ok
    assert reason == "terminal_replay_query_row_contract_failed"


def test_terminal_replay_output_differential_proof_passes_body_window_backend():
    torch.manual_seed(0)
    query = torch.randn(3, 2, 4, dtype=torch.float32)
    key = torch.randn(12, 2, 4, dtype=torch.float32)
    value = torch.randn(12, 2, 4, dtype=torch.float32)

    backend = _body_window_reference(query, key, value, envelope=4, query_start=7)
    proof = prove_segment_reuse_terminal_replay_output_semantics(
        query,
        key,
        value,
        backend,
        envelope_token_count=4,
        terminal_query_start=7,
        terminal_query_tokens=3,
        context_tokens=12,
    )

    assert proof["semantic_proven"] is True
    assert proof["semantic_reason"] == ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF
    assert proof["hidden_max_abs_diff"] <= 1e-6
    assert proof["all_context_max_abs_diff"] > 0.0


def test_terminal_replay_output_differential_proof_rejects_all_context_backend():
    torch.manual_seed(1)
    query = torch.randn(2, 2, 4, dtype=torch.float32)
    key = torch.randn(10, 2, 4, dtype=torch.float32)
    value = torch.randn(10, 2, 4, dtype=torch.float32)

    backend = _all_context_reference(query, key, value, query_start=6)
    proof = prove_segment_reuse_terminal_replay_output_semantics(
        query,
        key,
        value,
        backend,
        envelope_token_count=4,
        terminal_query_start=6,
        terminal_query_tokens=2,
        context_tokens=10,
    )

    assert proof["semantic_proven"] is False
    assert proof["semantic_reason"] == "terminal_replay_output_reference_mismatch"
    assert proof["hidden_max_abs_diff"] > 0.0


def test_terminal_replay_output_differential_supports_flat_tnd_and_gqa():
    torch.manual_seed(2)
    query = torch.randn(2, 4, 3, dtype=torch.float32)
    key = torch.randn(9, 2, 3, dtype=torch.float32)
    value = torch.randn(9, 2, 3, dtype=torch.float32)
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    backend = _body_window_reference(
        query,
        expanded_key,
        expanded_value,
        envelope=3,
        query_start=5,
    )

    proof = prove_segment_reuse_terminal_replay_output_semantics(
        query.reshape(2, 12),
        key.reshape(9, 6),
        value.reshape(9, 6),
        backend.reshape(2, 12),
        envelope_token_count=3,
        terminal_query_start=5,
        terminal_query_tokens=2,
        context_tokens=9,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=3,
    )

    assert proof["semantic_proven"] is True
    assert proof["semantic_reason"] == ASCEND_TERMINAL_REPLAY_OUTPUT_DIFFERENTIAL_PROOF


def test_terminal_replay_output_differential_fails_closed_for_invalid_window():
    query = torch.randn(1, 2, 4)
    key = torch.randn(8, 2, 4)
    value = torch.randn(8, 2, 4)
    backend = torch.zeros_like(query)

    proof = prove_segment_reuse_terminal_replay_output_semantics(
        query,
        key,
        value,
        backend,
        envelope_token_count=4,
        terminal_query_start=2,
        terminal_query_tokens=1,
        context_tokens=8,
    )

    assert proof["semantic_proven"] is False
    assert proof["semantic_reason"] == "terminal_replay_output_differential_failed"
    assert proof["error_type"] == "ValueError"


def test_materialize_terminal_replay_tensors_from_paged_block_table():
    query = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    key = torch.arange(4 * 4 * 1 * 3, dtype=torch.float32).reshape(4, 4, 1, 3)
    value = key + 1000
    block_table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int64)

    query_tnd, key_tnd, value_tnd, metadata = materialize_segment_reuse_terminal_replay_tensors(
        query,
        key,
        value,
        block_table=block_table,
        req_idx=1,
        context_tokens=6,
        block_size=4,
        terminal_query_tokens=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_size=3,
    )

    expected_key = torch.cat([key[1], key[3]], dim=0)[:6]
    expected_value = torch.cat([value[1], value[3]], dim=0)[:6]
    assert torch.equal(query_tnd, query)
    assert torch.equal(key_tnd, expected_key)
    assert torch.equal(value_tnd, expected_value)
    assert metadata["logical_kv_source"] == "triton-ascend-paged-cache-block-table"
    assert metadata["boundary_req_idx"] == 1


def test_materialize_terminal_replay_tensors_fails_closed_without_block_table():
    query = torch.randn(1, 2, 4)
    key = torch.randn(2, 4, 1, 4)
    value = torch.randn(2, 4, 1, 4)

    with pytest.raises(ValueError, match="block_table missing"):
        materialize_segment_reuse_terminal_replay_tensors(
            query,
            key,
            value,
            block_table=None,
            req_idx=0,
            context_tokens=4,
            block_size=4,
            terminal_query_tokens=1,
            num_query_heads=2,
            num_kv_heads=1,
            head_size=4,
        )


def _body_window_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    envelope: int,
    query_start: int,
) -> torch.Tensor:
    rows = []
    scale = query.shape[-1] ** -0.5
    for row in range(query.shape[0]):
        absolute_pos = query_start + row
        key_slice = key[envelope : absolute_pos + 1]
        value_slice = value[envelope : absolute_pos + 1]
        scores = (key_slice * query[row].unsqueeze(0)).sum(dim=-1).transpose(0, 1)
        weights = torch.softmax(scores * scale, dim=-1)
        rows.append((weights.transpose(0, 1).unsqueeze(-1) * value_slice).sum(dim=0))
    return torch.stack(rows)


def _all_context_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_start: int,
) -> torch.Tensor:
    rows = []
    scale = query.shape[-1] ** -0.5
    for row in range(query.shape[0]):
        absolute_pos = query_start + row
        key_slice = key[: absolute_pos + 1]
        value_slice = value[: absolute_pos + 1]
        scores = (key_slice * query[row].unsqueeze(0)).sum(dim=-1).transpose(0, 1)
        weights = torch.softmax(scores * scale, dim=-1)
        rows.append((weights.transpose(0, 1).unsqueeze(-1) * value_slice).sum(dim=0))
    return torch.stack(rows)
