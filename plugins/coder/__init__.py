"""Lexy AI — Self-Coding Workspace plugin."""

# Re-export the plugin entry-point at package level so the plugin
# loader's ``module.ClassName`` resolution finds it. We import lazily to
# avoid pulling the full plugin (and its httpx-based subprocess deps)
# at import-time during unit tests that only need the small modules.

__all__ = ["CoderPlugin"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "CoderPlugin":
        from .plugin import CoderPlugin
        return CoderPlugin
    raise AttributeError(name)
