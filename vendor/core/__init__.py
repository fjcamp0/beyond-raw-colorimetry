# Copyright 2024 F. Javier del Campo
# Licensed under the Apache License, Version 2.0

"""
core — shared library for raw-image-analysis.

Pipeline stages (s00–s04) and standalone scripts/ are thin CLIs over this
package. Entry points add the workflow root to sys.path before importing:

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import config, images, calibration
"""

__version__ = "1.0.0"
