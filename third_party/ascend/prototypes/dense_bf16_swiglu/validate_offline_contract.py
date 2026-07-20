"""Direct offline validator for the Triton-Ascend-owned AscendC prototype."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "b9f3fd4c2b795bcc67293757c7adaf032c624540a8f7d25d830657a8f77465c3"
ASCENDC_SOURCE_SHA256 = "5a98bab28feff4bf55d12b1902d5ff5372daf371897402d6e6e635b67dbc53f2"
HOST_REFERENCE_SHA256 = "5b9c2486266abecfb1ca797dbab1cce26138d84f9afd58a36feb8df903461f06"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(root: Path = ROOT) -> dict[str, object]:
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("prototype contract is not bound to the corrected protocol")
    if contract.get("package_state") != "REQUEST_ONLY_NOT_AUTHORIZED":
        raise ValueError("prototype package state changed")
    if contract.get("owner_repository") != "triton-ascend-hust" or contract.get("owner_path") != "third_party/ascend/prototypes/dense_bf16_swiglu":
        raise ValueError("prototype ownership changed")
    if (
        contract.get("status") != "stage0-static-prototype-not-cann-compiled"
        or contract.get("offline_only") is not True
        or contract.get("cann_compile_verified") is not False
        or contract.get("npu_correctness_verified") is not False
        or contract.get("performance_verified") is not False
        or contract.get("npu_execution_authorized") is not False
        or contract.get("reservation_request_generation_allowed") is not False
    ):
        raise ValueError("prototype overstates its offline verification status")
    if contract.get("dtype") != "bfloat16" or contract.get("accumulator_dtype") != "float32":
        raise ValueError("prototype must retain BF16 inputs/weights and FP32 accumulation")
    if contract.get("operation") != "down(silu(gate(x)) * up(x))":
        raise ValueError("prototype is not the independent dense SwiGLU carrier")
    if contract.get("natural_m") != [1, 4, 8, 16, 32]:
        raise ValueError("prototype natural-M cells changed")
    if contract.get("models") != [
        {"name": "Qwen2.5-7B", "hidden": 3584, "intermediate": 18944},
        {"name": "Qwen2.5-14B", "hidden": 5120, "intermediate": 13824},
    ]:
        raise ValueError("prototype model shape cells changed")
    space = contract.get("schedule_space")
    if not isinstance(space, Mapping):
        raise ValueError("prototype schedule space is missing")
    schedules = list(itertools.product(space.get("block_m", []), space.get("block_n", []), space.get("block_k", [])))
    if len(schedules) != 20 or len(set(schedules)) != 20 or space.get("candidate_count") != 20:
        raise ValueError("prototype schedule space must be the frozen 20-cell product")
    if space.get("block_m") != [1, 4, 8, 16, 32] or space.get("block_n") != [64, 128] or space.get("block_k") != [64, 128]:
        raise ValueError("prototype schedule values changed")
    if contract.get("abi") != ["x", "gate_weight", "up_weight", "down_weight", "y", "workspace", "tiling"]:
        raise ValueError("prototype ABI changed")
    return contract


def validate_ascendc_source(root: Path = ROOT) -> None:
    source_path = root / "ascendc_kernel.cpp"
    source = source_path.read_text(encoding="utf-8")
    if sha256_file(source_path) != ASCENDC_SOURCE_SHA256:
        raise ValueError("AscendC semantic prototype differs from its frozen source")
    required = (
        'extern "C" __global__ __aicore__ void dense_bf16_swiglu_proto',
        "GlobalTensor<bfloat16_t>", "gate_weight", "up_weight", "down_weight",
        "gate_sum", "up_sum", "expf(-gate_sum)", "output.SetValue",
        "inner_base += spec->block_n", "k_base += spec->block_k",
        "row_begin + spec->block_m",
    )
    if any(token not in source for token in required):
        raise ValueError("AscendC prototype lacks a required dense BF16 boundary")
    signature = re.search(
        r"dense_bf16_swiglu_proto\(\s*GM_ADDR x,\s*GM_ADDR gate_weight,\s*"
        r"GM_ADDR up_weight,\s*GM_ADDR down_weight,\s*GM_ADDR y,\s*"
        r"GM_ADDR workspace,\s*GM_ADDR tiling\)", source,
    )
    if signature is None:
        raise ValueError("AscendC prototype ABI does not match the contract")
    lowered = source.lower()
    if any(token in lowered for token in ("triton", "torch::", "quant", "moe", "grouped")):
        raise ValueError("prototype is not an independent dense BF16 implementation")


def source_manifest(root: Path = ROOT) -> dict[str, str]:
    required = ("README.md", "CMakeLists.txt", "contract.json", "host_reference.cpp", "ascendc_kernel.cpp", "validate_offline_contract.py")
    return {name: sha256_file(root / name) for name in required}


def build_and_test_host_contract(root: Path, build_dir: Path) -> None:
    """Build only the CPU reference; this never invokes CANN or a device backend."""
    validate_contract(root)
    validate_ascendc_source(root)
    if sha256_file(root / "host_reference.cpp") != HOST_REFERENCE_SHA256:
        raise ValueError("host oracle differs from its frozen source")
    if shutil.which("cmake") and shutil.which("ctest"):
        subprocess.run(("cmake", "-S", str(root), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"), check=True)
        subprocess.run(("cmake", "--build", str(build_dir), "--parallel", "2"), check=True)
        subprocess.run(("ctest", "--test-dir", str(build_dir), "--output-on-failure"), check=True)
        return
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("neither CMake nor a host C++ compiler is available")
    build_dir.mkdir(parents=True)
    executable = build_dir / "dense_bf16_swiglu_host_contract"
    subprocess.run((compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(root / "host_reference.cpp"), "-o", str(executable)), check=True)
    subprocess.run((str(executable),), check=True)


if __name__ == "__main__":
    validate_contract()
    validate_ascendc_source()
    print(json.dumps({"package_state": "REQUEST_ONLY_NOT_AUTHORIZED", "offline_only": True, "files": source_manifest()}, sort_keys=True))
