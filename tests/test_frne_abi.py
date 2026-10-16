"""A stale native layout must fall back before reading packed link data."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.testing as npt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.robot import frne


@pytest.mark.parametrize("version", [None, 0, 2])
def test_stale_native_abi_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    version: int | None,
) -> None:
    stale = ModuleType("roboticstoolbox._frne_c")

    def unexpected_call(*args: object) -> None:
        raise AssertionError("a stale native layout must not receive link data")

    stale.init = stale.frne = stale.delete = unexpected_call
    if version is not None:
        stale._ABI_VERSION = version
    monkeypatch.setitem(sys.modules, "roboticstoolbox._frne_c", stale)
    spec = importlib.util.spec_from_file_location("frne_stale", Path(frne.__file__))
    assert spec is not None and spec.loader is not None
    facade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facade)

    assert not facade._C_AVAILABLE
    assert facade.init(1, 0, []) is None
    facade.delete(None)


def test_unavailable_extension_executes_correct_flipped_python_dynamics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frne, "_C_AVAILABLE", False)
    robot = rtb.DHRobot(
        [rtb.PrismaticDH(flip=True, offset=0.4, m=2)],
        gravity=[0, 0, -9.81],
    )

    npt.assert_allclose(robot.rne([0.2], [0], [0]), [-19.62], atol=1e-13)
    assert robot._frne is None


@pytest.mark.skipif(not frne._C_AVAILABLE, reason="matching compiled _frne_c not built")
@pytest.mark.parametrize("size", [24, 26])
def test_new_native_abi_rejects_incompatible_packed_links(size: int) -> None:
    from roboticstoolbox import _frne_c

    with pytest.raises(ValueError, match="link data layout"):
        _frne_c.init(1, 0, np.zeros(size))
