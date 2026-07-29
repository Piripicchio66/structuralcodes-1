from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def test_requires_python_allows_python_314_patch_release():
    pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
    requires_python = pyproject['project']['requires-python']

    assert Version('3.14.4') in SpecifierSet(requires_python)
