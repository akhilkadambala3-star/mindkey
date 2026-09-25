"""Deterministic plain-text rendering (Phase 3.5).

Two pure renderers over the completed pipeline output:

- :func:`render_timeline` renders the investigation trace as the "AI
  Investigation Timeline" (reusing Phase 3.1's ``InvestigationTrace.timeline``),
  accepting engine events, critic events, or both.
- :func:`render_report_lines` / :func:`render_report` render a
  :class:`~investigation.agent.report.GroundedReport` as fixed plain-text
  sections.

Presentation only: no HTML, no dashboard, no I/O, no language model, no new
dependencies. Every line is produced from existing, already-grounded fields.
"""

from .report import GroundedReport
from .trace import InvestigationTrace

#: Version of the rendering shape.
TIMELINE_SCHEMA_VERSION = "1.0"

_TIMELINE_HEADING = "Investigation timeline:"


def render_timeline(events, *, clock=None) -> list[str]:
    """Render trace events as ordered ``HH:MM:SS -- <label> <subject>`` lines.

    Args:
        events: engine trace dicts/models, critic events, or both combined.
        clock: optional clock for the reconstructed trace (no events are added).

    Returns:
        The rendered lines, in trace order.
    """
    trace = InvestigationTrace.from_events(events, clock=clock)
    return trace.timeline()


def render_timeline_for(report, result, *, clock=None) -> list[str]:
    """Render the engine trace followed by the critic events as one timeline."""
    events = list(getattr(result.state, "trace", []) or [])
    events += list(getattr(report, "critic_events", []) or [])
    return render_timeline(events, clock=clock)


def render_report_lines(report: GroundedReport) -> list[str]:
    """Render a grounded report as fixed, plain-text sections.

    The section order is fixed so two runs over the same report render
    identically. No free text is introduced: each line is built from existing
    report fields.
    """
    lines = []
    lines.append(f"Conclusion: {report.conclusion.status} ({report.conclusion.basis})")
    lines.append(report.conclusion.statement)
    if report.conclusion.evidence_ids:
        lines.append(f"Conclusion evidence: {', '.join(report.conclusion.evidence_ids)}")

    lines.append("")
    lines.append("Summary:")
    lines.extend(f"- {line}" for line in report.summary)

    lines.append("")
    lines.append("Observations:")
    lines.extend(
        f"- [{claim.id}] {claim.statement} ({', '.join(claim.evidence_ids)})"
        for claim in report.observations
    )

    lines.append("")
    lines.append("Hypotheses:")
    lines.extend(
        f"- {claim.id} [{claim.status}] {claim.statement}"
        for claim in report.hypothesis_assessment
    )

    lines.append("")
    lines.append("Alternatives:")
    lines.extend(
        f"- {claim.id} [{claim.status}] {claim.statement}"
        for claim in report.alternatives
    )

    lines.append("")
    lines.append(f"Uncertainty: {report.uncertainty.level}")
    lines.extend(f"- {reason}" for reason in report.uncertainty.reasons)

    lines.append("")
    lines.append("Limitations:")
    lines.extend(f"- {limitation}" for limitation in report.limitations)

    if report.rejected_claims:
        lines.append("")
        lines.append("Rejected claims:")
        lines.extend(
            f"- {claim.subject} ({claim.reason})" for claim in report.rejected_claims
        )

    lines.append("")
    lines.append(f"Evidence digest: {report.link.evidence_digest}")
    lines.append(f"Engine trace digest: {report.link.engine_trace_digest}")
    lines.append(f"Critic trace digest: {report.link.critic_trace_digest}")
    lines.append(f"Trace events: {report.link.trace_event_count}")
    if report.link.stop_event_seq is not None:
        lines.append(f"Stop event seq: {report.link.stop_event_seq}")

    lines.append("")
    lines.append(report.disclaimer)
    return lines


def render_report(report: GroundedReport) -> str:
    """Render a grounded report as one plain-text string."""
    return "\n".join(render_report_lines(report))


def render(report: GroundedReport, result) -> str:
    """Render a report plus its investigation timeline as one plain-text string."""
    parts = render_report_lines(report)
    parts.append("")
    parts.append(_TIMELINE_HEADING)
    parts.extend(render_timeline_for(report, result))
    return "\n".join(parts)
