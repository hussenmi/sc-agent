"""Tests for the self-documenting artifact layer (scagent/core/artifact_docs.py)
and its adoption in the batch-effect diagnostic."""

import pandas as pd

from scagent.core import artifact_docs as ad


def _doc():
    return ad.ArtifactGroupDoc(
        group="demo_tool",
        title="Demo — how to read these files",
        overview="A demo artifact group.",
        params={"batch_key": "sample", "keys": ["a", "b"]},
        files=[
            ad.FileDoc(
                filename="scores.csv",
                purpose="Per-gene scores.",
                computation="AUC per gene.",
                columns={"gene": "Gene symbol.", "score": "AUC score."},
                how_to_read="Higher score = more discriminative.",
            )
        ],
    )


# --- column glossary ------------------------------------------------------------

def test_column_glossary_merges_frame_columns_with_descriptions():
    fdoc = _doc().files[0]
    df = pd.DataFrame({"gene": ["A", "B"], "score": [0.9, 0.1]})
    gloss = ad.column_glossary(fdoc, df)
    by_name = {g["name"]: g for g in gloss}
    assert by_name["gene"]["description"] == "Gene symbol."
    assert by_name["gene"]["dtype"] == "str"
    assert by_name["score"]["dtype"] == "float"


def test_column_glossary_flags_undocumented_and_missing_columns():
    fdoc = ad.FileDoc(filename="x.csv", purpose="p", columns={"documented_absent": "desc"})
    df = pd.DataFrame({"present_undoc": [1, 2]})
    gloss = ad.column_glossary(fdoc, df)
    by_name = {g["name"]: g for g in gloss}
    # A column in the frame but not authored is listed with a placeholder desc.
    assert by_name["present_undoc"]["description"] == "—"
    # An authored column absent from the frame is still surfaced (spec gap).
    assert by_name["documented_absent"]["description"] == "desc"
    assert by_name["documented_absent"]["dtype"] == "—"


def test_column_glossary_without_frame_uses_authored_columns():
    fdoc = _doc().files[0]
    gloss = ad.column_glossary(fdoc, None)
    assert {g["name"] for g in gloss} == {"gene", "score"}
    assert all(g["dtype"] == "—" for g in gloss)


# --- rendering ------------------------------------------------------------------

def test_render_readme_contains_marker_params_columns_and_interpretation():
    df = pd.DataFrame({"gene": ["A"], "score": [0.5]})
    text = ad.render_readme(_doc(), frames={"scores.csv": df})
    assert text.startswith(ad.DOC_MARKER)
    assert "## Parameters used" in text
    assert "`batch_key`: sample" in text
    assert "`keys`: a, b" in text
    assert "### `scores.csv`" in text
    assert "(1 rows)" in text
    assert "Higher score = more discriminative." in text
    assert "| `gene` | str | Gene symbol. |" in text
    assert ad.INTERPRETATION_HEADING in text
    assert ad.interpretation_is_empty(text) is True


def test_render_readme_opt_out_omits_interpretation_section():
    doc = _doc()
    doc.interpretation_required = False
    text = ad.render_readme(doc)
    assert ad.INTERPRETATION_HEADING not in text
    # An opt-out README is not considered "missing" an interpretation.
    assert ad.interpretation_is_empty(text) is False


def test_render_readme_escapes_pipes_in_descriptions():
    fdoc = ad.FileDoc(filename="x.csv", purpose="p", columns={"c": "a | b"})
    text = ad.render_readme(
        ad.ArtifactGroupDoc(group="g", title="t", overview="o", files=[fdoc])
    )
    assert "a \\| b" in text


# --- interpretation editing -----------------------------------------------------

def test_interpretation_is_empty_states():
    filled = ad.render_readme(_doc()).replace(
        ad.INTERPRETATION_PLACEHOLDER, "Clusters 4 and 12 are a batch split."
    )
    assert ad.interpretation_is_empty(filled) is False
    # No heading at all -> opted out, not empty.
    assert ad.interpretation_is_empty("no heading here") is False


def test_set_interpretation_replaces_placeholder():
    text = ad.render_readme(_doc())
    updated = ad.set_interpretation(text, "The samples are well mixed.")
    assert ad.interpretation_is_empty(updated) is False
    assert "The samples are well mixed." in updated
    assert ad.INTERPRETATION_PLACEHOLDER not in updated
    # Structural content above the section is preserved.
    assert "### `scores.csv`" in updated


def test_set_interpretation_appends_heading_when_absent():
    text = ad.DOC_MARKER + "\n\n# Title\n\nbody\n"
    updated = ad.set_interpretation(text, "Findings.")
    assert ad.INTERPRETATION_HEADING in updated
    assert "Findings." in updated


# --- finalize scan --------------------------------------------------------------

def test_scan_incomplete_interpretations(tmp_path):
    root = tmp_path
    # (1) required + empty -> flagged
    empty_dir = root / "artifacts" / "empty"
    empty_dir.mkdir(parents=True)
    ad.write_group_doc(empty_dir, _doc())
    # (2) required + filled -> not flagged
    filled_dir = root / "artifacts" / "filled"
    filled_dir.mkdir(parents=True)
    fp = ad.write_group_doc(filled_dir, _doc())
    fp.write_text(ad.set_interpretation(fp.read_text(), "Done."))
    # (3) opt-out README -> not flagged
    optout = _doc()
    optout.interpretation_required = False
    optout_dir = root / "artifacts" / "optout"
    optout_dir.mkdir(parents=True)
    ad.write_group_doc(optout_dir, optout)
    # (4) a hand-written README with no marker -> ignored
    (root / "README.md").write_text("# Not ours\n")

    missing = ad.scan_incomplete_interpretations(root)
    assert any("empty" in m for m in missing)
    assert not any("filled" in m for m in missing)
    assert not any("optout" in m for m in missing)


def test_artifact_column_metadata_shape():
    fdoc = _doc().files[0]
    meta = ad.artifact_column_metadata(fdoc)
    assert meta["purpose"] == "Per-gene scores."
    assert meta["computation"] == "AUC per gene."
    assert {c["name"] for c in meta["columns"]} == {"gene", "score"}


# --- batch diagnostic adoption --------------------------------------------------

def test_batch_diagnostic_write_outputs_emits_readme_and_metadata(tmp_path):
    from scagent.analysis.batch_diagnostic import (
        _batch_diagnostic_group_doc,
        _write_outputs,
    )

    df = pd.DataFrame(
        {
            "cluster": ["7", "12"],
            "sample": ["G8", "G3"],
            "n_cells": [200, 180],
            "n_cluster": [900, 850],
            "frac_of_cluster": [0.30, 0.28],
            "sample_baseline_frac": [0.10, 0.14],
            "enrichment": [3.0, 2.0],
        }
    )
    doc = _batch_diagnostic_group_doc({"batch_key": "dataset", "cluster_key": "leiden"})
    arts = _write_outputs(
        str(tmp_path),
        {"batch_diagnostic_sample_enriched_regions": df},
        group_doc=doc,
    )

    roles = {a["role"] for a in arts}
    assert "artifact" in roles and "artifact_readme" in roles

    csv_art = next(a for a in arts if a["role"] == "artifact")
    assert "columns" in csv_art["metadata"]
    assert any(c["name"] == "enrichment" for c in csv_art["metadata"]["columns"])

    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text()
    assert ad.DOC_MARKER in text
    assert "enrichment" in text
    assert ad.interpretation_is_empty(text) is True


def test_run_manager_complete_warns_on_missing_interpretation(tmp_path):
    from scagent.agent.run_manager import RunManager

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    group_dir = rm.run_dir / "artifacts" / "demo_tool"
    group_dir.mkdir(parents=True)
    ad.write_group_doc(group_dir, _doc())  # required + empty interpretation

    rm.complete(summary="done", request="req")

    assert any("interpretation" in w for w in rm.manifest.warnings)
    assert any("demo_tool" in w for w in rm.manifest.warnings)


def test_run_manager_complete_quiet_when_interpretation_present(tmp_path):
    from scagent.agent.run_manager import RunManager

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    group_dir = rm.run_dir / "artifacts" / "demo_tool"
    group_dir.mkdir(parents=True)
    path = ad.write_group_doc(group_dir, _doc())
    path.write_text(ad.set_interpretation(path.read_text(), "Findings recorded."))

    rm.complete(summary="done", request="req")

    assert not any("interpretation" in w for w in rm.manifest.warnings)


def test_annotate_artifact_group_writes_interpretation(tmp_path):
    import json

    from scagent.agent.run_manager import RunManager
    from scagent.agent.tools import process_tool_call

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    group_dir = rm.run_dir / "artifacts" / "demo_tool"
    group_dir.mkdir(parents=True)
    readme = ad.write_group_doc(group_dir, _doc())
    # Register the README so the handler can resolve it by group id.
    rm.add_artifact(
        {
            "path": str(readme),
            "role": "artifact_readme",
            "metadata": {"kind": "artifact_readme", "group": "demo_tool"},
        }
    )
    assert ad.interpretation_is_empty(readme.read_text()) is True

    rj, _ = process_tool_call(
        "annotate_artifact_group",
        {"group": "demo_tool", "interpretation": "Samples separate strongly by batch."},
        run_manager=rm,
    )
    result = json.loads(rj)
    assert result["status"] == "ok"
    text = readme.read_text()
    assert ad.interpretation_is_empty(text) is False
    assert "Samples separate strongly by batch." in text


def test_annotate_artifact_group_resolves_by_readme_path(tmp_path):
    import json

    from scagent.agent.run_manager import RunManager
    from scagent.agent.tools import process_tool_call

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    group_dir = rm.run_dir / "artifacts" / "demo_tool"
    group_dir.mkdir(parents=True)
    readme = ad.write_group_doc(group_dir, _doc())

    rj, _ = process_tool_call(
        "annotate_artifact_group",
        {"readme_path": str(readme), "interpretation": "Clean mixing."},
        run_manager=rm,
    )
    assert json.loads(rj)["status"] == "ok"
    assert "Clean mixing." in readme.read_text()


def test_annotate_artifact_group_errors_when_unresolvable(tmp_path):
    import json

    from scagent.agent.run_manager import RunManager
    from scagent.agent.tools import process_tool_call

    rm = RunManager(base_dir=str(tmp_path))
    rm.create()
    rj, _ = process_tool_call(
        "annotate_artifact_group",
        {"group": "nonexistent", "interpretation": "x"},
        run_manager=rm,
    )
    assert json.loads(rj)["status"] == "error"


def test_batch_diagnostic_group_doc_documents_all_five_files():
    doc = __import__(
        "scagent.analysis.batch_diagnostic", fromlist=["_batch_diagnostic_group_doc"]
    )._batch_diagnostic_group_doc({})
    names = {f.filename for f in doc.files}
    assert names == {
        "batch_diagnostic_sample_enriched_regions.csv",
        "batch_diagnostic_within_sample_degs.csv",
        "batch_diagnostic_population_pairs.csv",
        "batch_diagnostic_direct_pair_degs.csv",
        "batch_diagnostic_design_check.csv",
    }
    # every file has a purpose and at least one documented column
    for f in doc.files:
        assert f.purpose
        assert f.columns
