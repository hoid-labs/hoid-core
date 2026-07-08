import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_extras_map_matches_pyproject() -> None:
    """Every extra name referenced in _optional.EXTRAS_MAP must exist in pyproject.toml."""
    from hoid._optional import EXTRAS_MAP

    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    declared = set(data["project"]["optional-dependencies"].keys())
    for pkg_name, extra in EXTRAS_MAP.items():
        assert extra in declared, (
            f"_optional.EXTRAS_MAP maps '{pkg_name}' → '[{extra}]', "
            f"but '[{extra}]' is not declared in pyproject.toml. "
            f"Add it or fix the mapping."
        )
