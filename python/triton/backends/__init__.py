import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Type, TypeVar, Union
from types import ModuleType
from .driver import DriverBase
from .compiler import BaseBackend

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

T = TypeVar("T", bound=Union[BaseBackend, DriverBase])


def _find_concrete_subclasses(module: ModuleType, base_class: Type[T]) -> Type[T]:
    ret: list[Type[T]] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, base_class) and not inspect.isabstract(attr):
            ret.append(attr)
    if len(ret) == 0:
        raise RuntimeError(f"Found 0 concrete subclasses of {base_class} in {module}: {ret}")
    if len(ret) > 1:
        raise RuntimeError(f"Found >1 concrete subclasses of {base_class} in {module}: {ret}")
    return ret[0]


@dataclass(frozen=True)
class Backend:
    compiler: Type[BaseBackend]
    driver: Type[DriverBase]


def _discover_backends() -> dict[str, Backend]:
    backends = dict()
    requested = os.getenv("TRITON_CODEGEN_BACKENDS")
    requested_names = None
    if requested:
        requested_names = {name.strip() for name in requested.replace(",", ";").split(";") if name.strip()}
    for ep in entry_points().select(group="triton.backends"):
        if requested_names is not None and ep.name not in requested_names:
            continue
        compiler = importlib.import_module(f"{ep.value}.compiler")
        driver = importlib.import_module(f"{ep.value}.driver")
        backends[ep.name] = Backend(_find_concrete_subclasses(compiler, BaseBackend),  # type: ignore
                                    _find_concrete_subclasses(driver, DriverBase))  # type: ignore
    return backends


backends: dict[str, Backend] = _discover_backends()
