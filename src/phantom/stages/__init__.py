"""
Stage adapters: one module per pipeline area. Importing this package
registers every stage with :mod:`phantom.runner.stage`.

    preprocessing      collect genomes, metadata (labels), CheckV masks, plots
    features           build feature versions: import, merge, preprocess, extract
    features_analysis  diversity, ORF function summaries
    optimization       Optuna search over feature-engineering choices
    classification     single-modal, multimodal fusion, MDS
    tools              external tools through Snakemake, and their installation
"""

from phantom.stages import (  # noqa: F401
    classification,
    features,
    features_analysis,
    optimization,
    preprocessing,
    tools,
)
