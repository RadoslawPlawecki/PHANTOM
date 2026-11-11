"""
Stage adapters: one module per pipeline area. Importing this package
registers every stage with :mod:`phantom.runner.stage`.
"""

from phantom.stages import classification, features  # noqa: F401
