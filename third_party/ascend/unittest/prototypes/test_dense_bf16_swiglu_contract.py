from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PROTOTYPE = Path(__file__).resolve().parents[2] / "prototypes/dense_bf16_swiglu"
SPEC = importlib.util.spec_from_file_location("dense_bf16_swiglu_contract", PROTOTYPE / "validate_offline_contract.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load dense BF16 SwiGLU validator")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DenseBf16SwiGluContractTest(unittest.TestCase):
    def test_exact_static_contract_and_host_oracle(self) -> None:
        contract = VALIDATOR.validate_contract()
        VALIDATOR.validate_ascendc_source()
        self.assertEqual(contract["owner_repository"], "triton-ascend-hust")
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            VALIDATOR.build_and_test_host_contract(PROTOTYPE, Path(directory) / "build")

    def test_wrong_owner_or_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            copied = Path(directory) / "prototype"
            shutil.copytree(PROTOTYPE, copied)
            contract_path = copied / "contract.json"
            contract = json.loads(contract_path.read_text())
            contract["owner_repository"] = "vllm-ascend-hust"
            contract["npu_execution_authorized"] = True
            contract_path.write_text(json.dumps(contract))
            with self.assertRaises(ValueError):
                VALIDATOR.validate_contract(copied)

    def test_schedule_or_kernel_math_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            copied = Path(directory) / "prototype"
            shutil.copytree(PROTOTYPE, copied)
            contract_path = copied / "contract.json"
            contract = json.loads(contract_path.read_text())
            contract["schedule_space"]["block_n"] = [32, 256]
            contract_path.write_text(json.dumps(contract))
            with self.assertRaises(ValueError):
                VALIDATOR.validate_contract(copied)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            copied = Path(directory) / "prototype"
            shutil.copytree(PROTOTYPE, copied)
            kernel = copied / "ascendc_kernel.cpp"
            kernel.write_text(kernel.read_text().replace("result += silu", "result -= silu"))
            with self.assertRaises(ValueError):
                VALIDATOR.validate_ascendc_source(copied)


if __name__ == "__main__":
    unittest.main()
