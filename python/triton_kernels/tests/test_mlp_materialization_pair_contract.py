from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "triton_kernels/mlp_materialization_pair.py"


def test_pair_has_one_tmat_selected_schedule_identity() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert {"ScheduleIdentity", "tmat", "tfused", "_gate_up_materializing_kernel", "_gate_up_fused_kernel"} <= names
    assert 'value.get("selected_from_arm") != "Tmat"' in source
    assert "torch_npu.npu_swiglu(packed)" in source
    assert "_launch_fused(hidden, gate_up_weight, schedule, intermediate)" in source
    assert "torch.nn.functional.linear(activation, down_weight)" in source


def test_pair_kernels_share_candidate_axes_and_only_fused_epilogue_differs() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for token in ("BLOCK_M", "BLOCK_H", "BLOCK_I", "gate_acc", "up_acc", "tl.dot"):
        assert source.count(token) >= 2
    assert "packed_ptr" in source
    assert "activation = (gate_acc * tl.sigmoid(gate_acc)) * up_acc" in source
    assert "mock" not in source.lower()
    assert "dry_run" not in source.lower()
