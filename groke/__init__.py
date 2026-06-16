"""GROKE: Vision-free navigation-instruction evaluation via graph reasoning on OpenStreetMap.

This package is intentionally light on import-time side effects: submodules that
run a pipeline (e.g. ``groke.agents.*``) read data files at import and are meant to
be run as scripts (``python -m groke.<module>``), not imported from here.
"""
