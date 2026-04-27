"""ATS adapter interface. The architectural contract.

Discovery and form-fill are the two methods. Everything else (parsing,
scoring, cover letter writing) is ATS-agnostic and lives in stages/.

Design note: we deliberately do NOT have the adapter parse the job
listing into structured form — that's Qwen's job in the parse stage,
shared across all ATSes. Adapters return raw HTML.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class DiscoveredJob:
    """What an adapter returns from discover(). Just enough to dedupe and
    fetch the full posting later."""

    source: str
    source_id: str          # ATS-native id (e.g. Greenhouse job_id)
    url: str
    company: str
    title: str
    location: str | None
    raw_html: str           # the full posting HTML for Qwen to parse


@dataclass
class FieldFillRequest:
    """One field on the application form. Stages 4 produce these; the
    adapter executes them in stage 5."""

    label: str              # human label as seen on the form
    value: str              # the value to enter
    field_type: str         # 'text' | 'email' | 'phone' | 'select' | 'file' | 'textarea'
    required: bool = False


@dataclass
class FillResult:
    success: bool
    field_label: str
    error: str | None = None
    selector_used: str | None = None  # for the learning store


class ATSAdapter(ABC):
    """Subclass per ATS (greenhouse, lever, workday, icims, ashby).

    Implementations live in adapters/<name>.py and register themselves
    via REGISTRY at import time.
    """

    name: str  # subclass sets this

    @abstractmethod
    async def discover(self, board: str, **kwargs: Any) -> AsyncIterator[DiscoveredJob]:
        """Yield jobs from a board. Adapters know their ATS's URL pattern.

        For Greenhouse, `board` is a company slug. For Lever, similar.
        For Workday, it's a tenant + careers URL.
        """
        if False:  # pragma: no cover — abstract async generator
            yield  # type: ignore

    @abstractmethod
    async def fill_form(
        self,
        application_url: str,
        fields: list[FieldFillRequest],
        **kwargs: Any,
    ) -> list[FillResult]:
        """Fill the application form with the given field values. Does
        NOT click submit — that's the human gate. Returns one result
        per field for diagnostic purposes.
        """
        ...


# Adapter registry. Adapters self-register on import; the orchestrator
# imports the package and reads from here.
REGISTRY: dict[str, type[ATSAdapter]] = {}


def register(adapter_cls: type[ATSAdapter]) -> type[ATSAdapter]:
    """Decorator to add an adapter to the registry."""
    REGISTRY[adapter_cls.name] = adapter_cls
    return adapter_cls
