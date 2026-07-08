import pytest

from hoid._optional import EXTRAS_MAP, require


def test_require_returns_object_unchanged():
    sentinel = object()
    result = require("anything", sentinel)
    print(f"returned: {result!r}  is same object: {result is sentinel}")
    assert result is sentinel


def test_require_raises_when_none():
    with pytest.raises(ImportError):
        require("fastapi", None)


def test_require_error_contains_package_name():
    with pytest.raises(ImportError, match="fastapi"):
        require("fastapi", None)


def test_require_error_contains_extra_hint():
    with pytest.raises(ImportError, match=r"\[mcp\]"):
        require("fastapi", None)


def test_require_uses_extras_map_for_hint():
    "Every entry in EXTRAS_MAP produces an error message referencing the mapped extra."
    for pkg_name, extra in EXTRAS_MAP.items():
        try:
            require(pkg_name, None)
        except ImportError as e:
            print(f"  {pkg_name!r} -> [{extra}]: {e}")
            assert extra in str(e)
            continue
        raise AssertionError(f"require({pkg_name!r}, None) did not raise")


def test_require_unknown_package_falls_back_to_package_name():
    "A package not in EXTRAS_MAP uses its own name as the install hint."
    try:
        require("some-unknown-pkg", None)
    except ImportError as e:
        print(f"error message: {e}")
        assert "some-unknown-pkg" in str(e)
