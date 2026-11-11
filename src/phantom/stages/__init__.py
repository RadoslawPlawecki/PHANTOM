"""
Stage adapters: one module per pipeline area. Importing this package
registers every stage with :mod:`phantom.runner.stage`.
"""

from phantom.stages import (  # noqa: F401
    classification,
    features,
    features_build,
    optimization,
    preprocessing,
)
