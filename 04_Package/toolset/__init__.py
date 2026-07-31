"""
Levee-detection toolset: GIS, patch generation, model, inference and SFINCS tools.

Author: Jakub Zapletal
Date:   2026-04-02
"""

from . import gis  # noqa: F401

# patches / models / inference / sfincs are imported explicitly by the scripts
# so that an environment does not need the dependencies of tools it never uses
