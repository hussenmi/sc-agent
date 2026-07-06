"""Self-documenting artifact outputs.

Any tool that writes files into an ``artifacts/<group>/`` (or figure) folder
should also drop a ``README.md`` that explains, per file: what was computed, why,
how to read it, and what each column means — plus a machine-readable column
glossary stamped onto the manifest artifact record so the agent can recall column
meanings in later steps without re-reading the file.

Division of labour (see AGENTS.md — the harness never bakes in domain knowledge,
and interpretation is the model's job):

  * STRUCTURE (purpose, computation, column glossary, parameters) is authored
    ONCE next to the producing tool and rendered here, deterministically. It
    describes what the tool computes — not a biological conclusion.
  * INTERPRETATION ("what the results tell us for THIS dataset") is left to the
    model, which fills the trailing section via the ``annotate_artifact_group``
    tool. Until it does, the section holds a placeholder and the finalize check
    surfaces a soft warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A hidden marker on the first line of every README we generate, so the finalize
# scanner only ever inspects our own docs and never a hand-written README.
DOC_MARKER = "<!-- scagent-artifact-doc -->"

INTERPRETATION_HEADING = "## Interpretation (dataset-specific)"
INTERPRETATION_PLACEHOLDER = (
    "_Pending — the analysis agent fills this in after reviewing the results "
    "(via `annotate_artifact_group`)._"
)


@dataclass
class FileDoc:
    """Documentation for a single output file within a group.

    ``columns`` maps column name -> human description. For non-tabular files
    (figures, JSON), leave it empty and use ``how_to_read`` to explain the file.
    """

    filename: str
    purpose: str
    computation: str = ""
    columns: dict[str, str] = field(default_factory=dict)
    how_to_read: str = ""


@dataclass
class ArtifactGroupDoc:
    """Documentation for a folder of related artifacts (one README per group).

    ``interpretation_required`` controls whether the README carries an
    Interpretation section the model must fill (and the finalize check enforces).
    Set it False for reference docs whose dataset-specific findings already live
    in a companion report (e.g. structure QC's markdown summary), so we don't
    demand a redundant annotation on every pass.
    """

    group: str
    title: str
    overview: str
    files: list[FileDoc] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    interpretation_required: bool = True


# --- dtype labelling ------------------------------------------------------------

def _dtype_label(series: Any) -> str:
    """A friendly type label for a pandas Series/dtype, best-effort."""
    try:
        import pandas as pd
        from pandas.api import types as pdt

        dtype = series.dtype if hasattr(series, "dtype") else series
        if pdt.is_bool_dtype(dtype):
            return "bool"
        if pdt.is_integer_dtype(dtype):
            return "int"
        if pdt.is_float_dtype(dtype):
            return "float"
        if isinstance(dtype, pd.CategoricalDtype):
            return "category"
        if pdt.is_object_dtype(dtype):
            return "str"
        return str(dtype)
    except Exception:
        return "—"


def _resolve_frame(frames: dict[str, Any] | None, filename: str) -> Any:
    """Look up a DataFrame for ``filename`` by full name or stem (no extension)."""
    if not frames:
        return None
    if filename in frames:
        return frames[filename]
    stem = Path(filename).stem
    return frames.get(stem)


def column_glossary(doc: FileDoc, frame: Any = None) -> list[dict[str, str]]:
    """Merge authored column descriptions with the frame's actual columns.

    Actual columns (name + dtype) come from the frame when available, so the
    glossary never drifts from what was written; descriptions come from the
    authored ``doc.columns``. Columns present in the frame but undocumented are
    still listed (description "—"); documented columns absent from the frame are
    appended after, so a spec gap is visible either way.
    """
    entries: list[dict[str, str]] = []
    seen: set = set()
    if frame is not None and hasattr(frame, "columns"):
        for name in list(frame.columns):
            key = str(name)
            entries.append(
                {
                    "name": key,
                    "dtype": _dtype_label(frame[name]),
                    "description": doc.columns.get(key, "—"),
                }
            )
            seen.add(key)
    for name, desc in doc.columns.items():
        if str(name) not in seen:
            entries.append({"name": str(name), "dtype": "—", "description": desc})
    return entries


# --- rendering ------------------------------------------------------------------

def _format_param(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value)


def render_readme(doc: ArtifactGroupDoc, frames: dict[str, Any] | None = None) -> str:
    """Render the group README markdown. ``frames`` is an optional mapping of
    filename (or stem) -> DataFrame used to fill real column names and dtypes.
    """
    lines: list[str] = [DOC_MARKER, "", f"# {doc.title}", "", doc.overview.strip(), ""]
    lines.append(
        f"_Produced by `{doc.group}`. This README is auto-generated; the "
        f"**Interpretation** section at the end is written by the analysis agent._"
    )
    lines.append("")

    if doc.params:
        lines.append("## Parameters used")
        lines.append("")
        for key in doc.params:
            lines.append(f"- `{key}`: {_format_param(doc.params[key])}")
        lines.append("")

    lines.append("## Files")
    lines.append("")
    for fdoc in doc.files:
        frame = _resolve_frame(frames, fdoc.filename)
        n_rows = None
        if frame is not None and hasattr(frame, "shape"):
            try:
                n_rows = int(frame.shape[0])
            except Exception:
                n_rows = None
        heading = f"### `{fdoc.filename}`"
        if n_rows is not None:
            heading += f"  ({n_rows} rows)"
        lines.append(heading)
        lines.append("")
        lines.append(f"**What it is.** {fdoc.purpose.strip()}")
        if fdoc.computation.strip():
            lines.append("")
            lines.append(f"**How it was computed.** {fdoc.computation.strip()}")
        if fdoc.how_to_read.strip():
            lines.append("")
            lines.append(f"**How to read it.** {fdoc.how_to_read.strip()}")

        glossary = column_glossary(fdoc, frame)
        if glossary:
            lines.append("")
            lines.append("| Column | Type | Meaning |")
            lines.append("| --- | --- | --- |")
            for entry in glossary:
                meaning = entry["description"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{entry['name']}` | {entry['dtype']} | {meaning} |")
        lines.append("")

    if doc.interpretation_required:
        lines.append(INTERPRETATION_HEADING)
        lines.append("")
        lines.append(INTERPRETATION_PLACEHOLDER)
        lines.append("")
    return "\n".join(lines)


def group_readme_path(output_dir: Any) -> Path:
    return Path(output_dir) / "README.md"


def write_group_doc(
    output_dir: Any, doc: ArtifactGroupDoc, frames: dict[str, Any] | None = None
) -> Path:
    """Render and write ``README.md`` into ``output_dir``; return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = group_readme_path(out)
    path.write_text(render_readme(doc, frames))
    return path


def artifact_column_metadata(doc: FileDoc, frame: Any = None) -> dict[str, Any]:
    """Machine-readable metadata for a single file's manifest artifact record."""
    return {
        "purpose": doc.purpose,
        "computation": doc.computation,
        "columns": column_glossary(doc, frame),
    }


# --- interpretation section -----------------------------------------------------

def _split_on_interpretation(text: str) -> tuple[str, str | None]:
    """Return (head_including_heading, body_after_heading). body is None if the
    heading is absent."""
    idx = text.find(INTERPRETATION_HEADING)
    if idx == -1:
        return text, None
    head_end = idx + len(INTERPRETATION_HEADING)
    return text[:head_end], text[head_end:]


def interpretation_is_empty(readme_text: str) -> bool:
    """True if the README has an Interpretation heading whose body is blank or
    still the placeholder. A README that omits the heading entirely opted out of
    requiring interpretation and is NOT considered empty."""
    _, body = _split_on_interpretation(readme_text)
    if body is None:
        return False
    stripped = body.strip()
    if not stripped:
        return True
    return stripped == INTERPRETATION_PLACEHOLDER.strip()


def set_interpretation(readme_text: str, interpretation: str) -> str:
    """Return the README with its Interpretation section replaced by
    ``interpretation``. If the heading is absent, append the section."""
    head, body = _split_on_interpretation(readme_text)
    new_body = "\n\n" + interpretation.strip() + "\n"
    if body is None:
        sep = "" if head.endswith("\n") else "\n"
        return head + sep + "\n" + INTERPRETATION_HEADING + new_body
    return head + new_body


def scan_incomplete_interpretations(run_dir: Any) -> list[str]:
    """Return run-relative paths of generated READMEs whose Interpretation
    section is still empty. Only inspects READMEs carrying ``DOC_MARKER``."""
    root = Path(run_dir)
    missing: list[str] = []
    if not root.exists():
        return missing
    for path in sorted(root.rglob("README.md")):
        try:
            text = path.read_text()
        except Exception:
            continue
        if DOC_MARKER not in text:
            continue
        if interpretation_is_empty(text):
            try:
                missing.append(str(path.relative_to(root)))
            except Exception:
                missing.append(str(path))
    return missing
