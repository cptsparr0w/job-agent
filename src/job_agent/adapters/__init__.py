"""ATS adapters. Each ATS gets one adapter implementing the interface
in `base.py`. Discovery and form-fill go through these.

The interface is deliberately small. Adding a new ATS = subclassing
`ATSAdapter` and implementing `discover()` and `fill_form()`. Qwen can
fill these in given the interface and one example.
"""

from .base import ATSAdapter, DiscoveredJob, FieldFillRequest, FillResult

__all__ = ["ATSAdapter", "DiscoveredJob", "FieldFillRequest", "FillResult"]
