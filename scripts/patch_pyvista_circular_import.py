#!/usr/bin/env python3
"""
Patch PyVista to fix circular import: AttributeError: partially initialized module
'pyvista' has no attribute '_plot'.

Run: python scripts/patch_pyvista_circular_import.py

This modifies the installed pyvista package in site-packages. Re-run after
upgrading pyvista.
"""

import importlib.util
import re
import sys
from pathlib import Path


def find_pyvista_path() -> Path:
    """Find pyvista package path without importing it."""
    spec = importlib.util.find_spec("pyvista")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyvista not found. Install with: pip install pyvista")
    return Path(spec.submodule_search_locations[0])


def patch_file(filepath: Path) -> bool:
    """Replace 'plot = pv._plot.plot' or 'plot = pyvista._plot.plot' with lazy method."""
    content = filepath.read_text(encoding="utf-8")
    # Match: optional indent + plot = (pv|pyvista)._plot.plot
    pattern = re.compile(r"^(\s*)plot\s*=\s*(?:pv|pyvista)\._plot\.plot\s*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return False
    indent = match.group(1)
    # Same indent for def, +4 for method body
    new_block = (
        f"{indent}def plot(self, *args, **kwargs):\n"
        f"{indent}    from pyvista import _plot\n"
        f"{indent}    return _plot.plot(self, *args, **kwargs)"
    )
    content = pattern.sub(new_block, content)
    filepath.write_text(content, encoding="utf-8")
    return True


def main():
    pv_path = find_pyvista_path()
    print(f"PyVista path: {pv_path}")

    files_to_patch = [
        pv_path / "core" / "dataset.py",
        pv_path / "core" / "composite.py",
    ]

    patched = 0
    for fp in files_to_patch:
        if fp.exists():
            if patch_file(fp):
                print(f"  Patched: {fp.relative_to(pv_path.parent)}")
                patched += 1
            else:
                print(f"  Skip (no match): {fp.relative_to(pv_path.parent)}")
        else:
            print(f"  Not found: {fp}")

    if patched > 0:
        print("\nDone. Try: import pyvista as pv")
    else:
        print("\nNo files patched. PyVista version may differ.")
        sys.exit(1)


if __name__ == "__main__":
    main()
