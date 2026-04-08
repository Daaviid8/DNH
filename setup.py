"""
setup.py — optionally build Cython extensions.

The Cython extension is compiled when:
  - Cython is installed, AND
  - A C compiler is available.

If either is missing, the package installs in pure-Python mode
(slower, but fully functional).

Usage
-----
    pip install -e .                    # editable install, Cython if available
    pip install -e . --no-build-isolation
    python setup.py build_ext --inplace # explicit compile
    DNHDT_PURE_PYTHON=1 pip install -e . # skip Cython even if available
"""

import os
import sys
from setuptools import setup

_SKIP = os.environ.get("DNHDT_PURE_PYTHON", "0") == "1"


def _make_extensions():
    if _SKIP:
        return []
    try:
        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Extension

        ext = Extension(
            name="dnhdt._criterion",
            sources=["src/dnhdt/_criterion.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c",
        )
        return cythonize(
            [ext],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "nonecheck": False,
                "initializedcheck": False,
            },
            annotate=False,
        )
    except Exception as e:
        print(f"[dnhdt] Cython extension NOT built ({e}). "
              f"Using pure-Python fallback.", file=sys.stderr)
        return []


setup(ext_modules=_make_extensions())
