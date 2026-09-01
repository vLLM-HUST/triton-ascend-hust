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


def test_ascend_entrypoint_drops_unrelated_builtin_backends():
    module = _load_setup_ascend()

    class Backend:
        def __init__(self, name, is_external=False):
            self.name = name
            self.is_external = is_external

    class Installer:
        @staticmethod
        def prepare(name):
            return Backend(name)

    class SetupModule:
        BackendInstaller = Installer
        backends = [Backend("nvidia"), Backend("amd"), Backend("custom", True)]

    module._select_ascend_backends(SetupModule)

    assert [backend.name for backend in SetupModule.backends] == ["ascend", "custom"]
