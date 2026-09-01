import importlib.util
from pathlib import Path

from packaging.requirements import Requirement


def _load_setup_ascend():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "triton_ascend_setup_metadata", root / "setup_ascend.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_requirement_supports_legacy_and_modern_serving_stacks():
    module = _load_setup_ascend()
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, module._get_install_requirements())
    }

    numpy = requirements["numpy"].specifier
    assert "1.26.4" in numpy
    assert "2.2.6" in numpy
    assert "2.3.0" not in numpy
