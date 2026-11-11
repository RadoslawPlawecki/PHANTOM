"""
Non-interactive execution layer: run files (``runs/*.toml``) describe *what*
to compute, ``config.toml`` describes *where* things live, and every pipeline
step is a registered :class:`~phantom.runner.stage.Stage` with a typed
``Params`` dataclass. See ``phantom --help``.
"""
