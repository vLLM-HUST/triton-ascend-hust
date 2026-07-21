from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "prototypes/dense_bf16_materialization_pair"
SPEC = importlib.util.spec_from_file_location("pair_contract", ROOT / "validate_offline_contract.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_independent_pair_static_and_host_contract() -> None:
    contract = MODULE.validate(ROOT)
    assert contract["same_schedule_identity_required"] is True
    with tempfile.TemporaryDirectory() as directory:
        MODULE.build_host(ROOT, Path(directory) / "build")
