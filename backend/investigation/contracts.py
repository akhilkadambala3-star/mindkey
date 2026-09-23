"""The ML <-> Agent Pydantic contract (Phase 2).

``BehavioralEvidence`` is the structured object the investigation agent will
consume. It is assembled deterministically from existing stored data; it never
contains typed content, and it never asserts a medical condition.

Safety rules encoded by these models:

- Every value that may be missing is optional and defaults to ``None``; a
  missing value is represented explicitly, never invented.
- ``persistence.status`` can be ``"insufficient_data"``; it is never allowed to
  claim persistence when the history is too small.
- Anomaly numbers are exposed only as numbers (``anomaly_score`` /
  ``is_anomaly``); there is no diagnosis or cause field anywhere.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Direction = Literal["increase", "decrease", "unknown"]
PersistenceStatus = Literal[
    "insufficient_data",
    "stable",
    "recent_variation",
    "persistent_change",
    "unknown",
]
UncertaintyLevel = Literal["low", "moderate", "high", "unknown"]


class TriggerInfo(BaseModel):
    """What caused this evidence package to be built."""

    source: str = Field(default="typing_session")
    session_id: str | None = None
    session_start: datetime | None = None


class CanonicalSession(BaseModel):
    """A single typing session in canonical units.

    All feature values are optional: a partial or invalid stored row yields
    ``is_valid=False`` with ``None`` for missing values rather than a guess.
    """

    session_id: str | None = None
    user_id: str | None = None
    session_start: datetime | None = None
    session_end: datetime | None = None
    is_valid: bool = False
    typing_speed_cpm: float | None = None
    dwell_mean_s: float | None = None
    flight_mean_s: float | None = None
    correction_rate: float | None = None
    rhythm_variability_s: float | None = None
    pause_count: int | None = None


class FeatureStats(BaseModel):
    """Summary statistics for one canonical feature over a set of sessions."""

    key: str
    label: str
    canonical_unit: str
    available: bool
    sample_count: int = 0
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


class SignalEvidence(BaseModel):
    """One behavioral signal: a measured value plus its historical reference."""

    key: str
    label: str
    canonical_unit: str
    available: bool
    value: float | None = None
    baseline_value: float | None = None
    absolute_change: float | None = None
    relative_change: float | None = None
    direction: Direction = "unknown"


class WindowSummary(BaseModel):
    """Session counts and time bounds for one analysis window."""

    label: str
    window_days: int | None = None
    session_count: int = 0
    valid_session_count: int = 0
    start: datetime | None = None
    end: datetime | None = None


class PersistenceAssessment(BaseModel):
    """A conservative, deterministic hint about persistence.

    This is a heuristic on top of the evidence, not a conclusion. When there is
    not enough valid history it is forced to ``"insufficient_data"``.
    """

    status: PersistenceStatus = "unknown"
    supporting_session_count: int = 0
    signals_moved: int = 0
    note: str = ""


class TemporalAnalysis(BaseModel):
    """Recent vs. historical window summaries and a persistence hint."""

    recent: WindowSummary
    baseline: WindowSummary
    persistence: PersistenceAssessment
    recent_feature_stats: list[FeatureStats] = Field(default_factory=list)
    baseline_feature_stats: list[FeatureStats] = Field(default_factory=list)


class DataQuality(BaseModel):
    """How much usable data actually exists."""

    total_sessions: int = 0
    valid_sessions: int = 0
    invalid_sessions: int = 0
    sessions_without_timestamp: int = 0
    model_minimum_sessions: int = 0
    meets_model_minimum: bool = False
    issues: list[str] = Field(default_factory=list)


class ContextEvidence(BaseModel):
    """User-provided contextual factors.

    No backend store for check-ins or symptoms exists yet, so this is reported
    honestly as unavailable. It is never fabricated.
    """

    source: str = "unavailable"
    checkins: list[dict] = Field(default_factory=list)
    symptoms: dict | None = None
    note: str = ""


class Uncertainty(BaseModel):
    """How much confidence the evidence supports, and why."""

    level: UncertaintyLevel = "unknown"
    reasons: list[str] = Field(default_factory=list)


class BehavioralEvidence(BaseModel):
    """The structured evidence package handed to the investigation layer."""

    schema_version: str = SCHEMA_VERSION
    user_id: str
    generated_at: datetime
    trigger: TriggerInfo
    signals: list[SignalEvidence] = Field(default_factory=list)
    temporal_analysis: TemporalAnalysis
    data_quality: DataQuality
    context: ContextEvidence
    uncertainty: Uncertainty
    limitations: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
