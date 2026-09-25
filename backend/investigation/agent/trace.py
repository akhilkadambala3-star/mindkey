"""Investigation trace infrastructure (Phase 3.1).

The trace is the agent's audit log: an ordered, inspectable list of what the
investigation did. It exists so that a later phase can render an
"AI Investigation Timeline" and so that tests can assert *behaviour* (which
nodes ran, in what order, which tools were called, which decisions were made)
instead of only inspecting the final report.

What the trace is allowed to contain
------------------------------------
- node names, event types, tool names
- a **fingerprint** (truncated SHA-256) of a tool call's arguments -- never the
  raw argument values, so no user identifiers or rows leak into a log
- evidence ids (``E1``, ``E2``, ...) that point back into the evidence registry
- short, system-authored labels

What the trace must never contain
---------------------------------
- stored session rows or any feature value copied out of one
- typed content, keystrokes, passwords, or any other user-authored text
- secrets or credentials

There is deliberately no free-form logging API: ``detail`` is a short
system-authored label passed by the caller, not a place to dump data.

Timestamps are the only non-deterministic part of a trace. ``InvestigationTrace``
accepts an injectable ``clock`` so tests (and the deterministic demo) can pin
them.

This module has no dependencies beyond Pydantic and the standard library, and it
performs no I/O of any kind.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable, Iterable, Literal

from pydantic import BaseModel, Field

#: The closed vocabulary of trace event types. Pydantic rejects anything else.
TraceEventType = Literal[
    "node_enter",
    "node_exit",
    "tool_call",
    "tool_result",
    "evidence_added",
    "hypothesis_updated",
    "decision",
    "critic_rejection",
    "stop",
    "state_change",
]

#: Fixed, system-authored labels used when rendering the timeline.
EVENT_LABELS = {
    "node_enter": "Entered",
    "node_exit": "Completed",
    "tool_call": "Called tool",
    "tool_result": "Tool result",
    "evidence_added": "Registered evidence",
    "hypothesis_updated": "Updated hypothesis",
    "decision": "Decision",
    "critic_rejection": "Rejected unsupported claim",
    "stop": "Stopped",
    "state_change": "State updated",
}

#: Length of a truncated SHA-256 argument fingerprint (hex characters).
FINGERPRINT_LENGTH = 16


def _utcnow():
    return datetime.now(timezone.utc)


def args_fingerprint(tool, **args):
    """Return a short, stable, order-independent fingerprint of tool arguments.

    The fingerprint is a hash, so the original values cannot be recovered from
    the trace. It is used for tracing and for suppressing duplicate tool calls
    (same tool + same arguments must never run twice).
    """
    payload = json.dumps({"tool": tool, "args": args}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


class TraceEvent(BaseModel):
    """One recorded step of an investigation."""

    seq: int = Field(ge=1)
    ts: datetime
    node: str = Field(min_length=1)
    event_type: TraceEventType
    tool: str | None = None
    args_fingerprint: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    detail: str = ""


class InvestigationTrace:
    """An append-only, ordered trace of an investigation.

    Args:
        clock: optional zero-argument callable returning an aware ``datetime``.
            Defaults to ``datetime.now(timezone.utc)``; tests inject a fixed
            clock to make traces fully deterministic.
    """

    def __init__(self, clock: Callable[[], datetime] | None = None):
        self._events: list[TraceEvent] = []
        self._clock = clock or _utcnow

    @classmethod
    def from_events(cls, events, clock: Callable[[], datetime] | None = None):
        """Build a trace pre-loaded with existing events (Phase 3.4).

        A later phase (the critic) needs to **continue the same investigation
        trace** instead of starting a second one. The supplied events keep their
        ``seq`` values, so the next recorded event continues the sequence.

        Args:
            events: an iterable of :class:`TraceEvent` objects or their dict
                form (as produced by :meth:`to_dicts`).
            clock: optional zero-argument clock for subsequently recorded
                events.
        """
        trace = cls(clock=clock)
        trace._events = [
            event if isinstance(event, TraceEvent) else TraceEvent(**event)
            for event in (events or ())
        ]
        return trace

    # -- recording ---------------------------------------------------------

    def record(
        self,
        node: str,
        event_type: TraceEventType,
        *,
        tool: str | None = None,
        args: dict | None = None,
        evidence_ids: Iterable[str] = (),
        detail: str = "",
    ) -> TraceEvent:
        """Append one event and return it.

        ``args`` is fingerprinted, never stored verbatim.
        """
        event = TraceEvent(
            seq=len(self._events) + 1,
            ts=self._clock(),
            node=node,
            event_type=event_type,
            tool=tool,
            args_fingerprint=None if args is None else args_fingerprint(tool or "", **args),
            evidence_ids=list(evidence_ids),
            detail=detail,
        )
        self._events.append(event)
        return event

    def record_enter(self, node, *, detail=""):
        return self.record(node, "node_enter", detail=detail)

    def record_exit(self, node, *, detail=""):
        return self.record(node, "node_exit", detail=detail)

    def record_tool_call(self, node, tool, args=None, *, detail=""):
        return self.record(node, "tool_call", tool=tool, args=args, detail=detail)

    def record_tool_result(self, node, tool, *, evidence_ids=(), detail=""):
        return self.record(
            node, "tool_result", tool=tool, evidence_ids=evidence_ids, detail=detail
        )

    def record_evidence(self, node, evidence_ids, *, detail=""):
        return self.record(
            node, "evidence_added", evidence_ids=evidence_ids, detail=detail
        )

    def record_hypothesis_update(self, node, *, evidence_ids=(), detail=""):
        return self.record(
            node, "hypothesis_updated", evidence_ids=evidence_ids, detail=detail
        )

    def record_decision(self, node, decision, *, detail=""):
        return self.record(node, "decision", detail=detail or decision)

    def record_critic_rejection(self, node, *, evidence_ids=(), detail=""):
        return self.record(
            node, "critic_rejection", evidence_ids=evidence_ids, detail=detail
        )

    def record_state_change(self, node, *, detail=""):
        return self.record(node, "state_change", detail=detail)

    def record_stop(self, node, stop_reason, *, detail=""):
        return self.record(node, "stop", detail=detail or stop_reason)

    # -- reading -----------------------------------------------------------

    @property
    def events(self) -> list[TraceEvent]:
        """A copy of the recorded events, in order."""
        return list(self._events)

    def to_dicts(self) -> list[dict]:
        """The trace as JSON-serializable dictionaries."""
        return [event.model_dump(mode="json") for event in self._events]

    def timeline(self) -> list[str]:
        """Render the trace as ``HH:MM:SS -- <label> <subject>`` lines.

        Used by the (later) demo to display the AI Investigation Timeline.
        """
        lines = []
        for event in self._events:
            label = EVENT_LABELS.get(event.event_type, event.event_type)
            subject = event.tool or event.node
            text = f"{event.ts:%H:%M:%S} \u2014 {label} {subject}"
            if event.detail:
                text = f"{text}: {event.detail}"
            lines.append(text)
        return lines

    def __len__(self):
        return len(self._events)

    def __iter__(self):
        return iter(self._events)
