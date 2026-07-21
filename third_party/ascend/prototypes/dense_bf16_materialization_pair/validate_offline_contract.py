from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def validate(root: Path = ROOT) -> dict[str, object]:
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract["owner_repository"] != "triton-ascend-hust":
        raise ValueError("independent pair has the wrong owner")
    if contract["status"] != "offline-qualified-awaiting-grant-bound-cann-build-and-correctness":
        raise ValueError("independent pair overstates its online status")
    if contract["execution_authorized"] is not False or contract["performance_verified"] is not False:
        raise ValueError("offline contract may not authorize or claim performance")
    if contract["offline_contract_qualified"] is not True or contract["timing_eligible_after_grant_compile_and_correctness"] is not True:
        raise ValueError("timing qualification gate changed")
    space = contract["schedule_space"]
    schedules = list(itertools.product(space["block_m"], space["block_h"], space["block_i"]))
    if len(schedules) != 27 or len(set(schedules)) != 27 or space["candidate_count"] != 27:
        raise ValueError("independent candidate space changed")
    source = (root / "ascendc_pair_kernel.cpp").read_text(encoding="utf-8")
    required = (
        "dense_bf16_pair_tmat", "dense_bf16_pair_tfused", "DenseBf16GateUpPair<true>",
        "DenseBf16GateUpPair<false>", "schedule_identity", "SetSingleShape", "SetTensorA",
        "SetTensorB", "Iterate", "GetTensorC<false>", "DataCopyPad", "Sigmoid", "Mul",
        "tail_m", "tail_h", "tail_i", "GetBlockIdx",
    )
    if any(token not in source for token in required):
        raise ValueError("AscendC pair lacks a required tiled primitive")
    if "GetValue(" in source or "SetValue(" in source:
        raise ValueError("scalar GetValue/SetValue skeleton is timing-ineligible")
    return contract


def build_host(root: Path, build: Path) -> None:
    validate(root)
    if shutil.which("cmake") and shutil.which("ctest"):
        subprocess.run(("cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"), check=True)
        subprocess.run(("cmake", "--build", str(build), "--parallel", "2"), check=True)
        subprocess.run(("ctest", "--test-dir", str(build), "--output-on-failure"), check=True)
        return
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("neither CMake nor a host C++ compiler is available")
    build.mkdir(parents=True)
    executable = build / "dense_bf16_materialization_pair_host"
    subprocess.run((compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(root / "host_reference.cpp"), "-o", str(executable)), check=True)
    subprocess.run((str(executable),), check=True)


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
