"""Lexy AI — Deep-Research plugin (multi-step web research with citations)."""

__all__ = ["DeepResearchPlugin"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "DeepResearchPlugin":
        from .plugin import DeepResearchPlugin
        return DeepResearchPlugin
    raise AttributeError(name)
