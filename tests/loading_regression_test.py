import importlib.util

import anndata as ad
import numpy as np
import pandas as pd

from scagent.agent.tools import get_tools
from scagent.core import concat_datasets, discover_data_inputs


def _dataset(prefix: str, n_cells: int) -> ad.AnnData:
    return ad.AnnData(
        np.ones((n_cells, 3)),
        obs=pd.DataFrame(index=[f"{prefix}_cell_{index}" for index in range(n_cells)]),
        var=pd.DataFrame(
            {"feature_type": ["Gene Expression"] * 3},
            index=["GENE1", "GENE1", "GENE2"],
        ),
    )


def test_analysis_context_import_does_not_require_optional_modules():
    from scagent.analysis.context import infer_biological_context

    assert callable(infer_biological_context)


def test_missing_optional_analysis_tools_are_not_advertised():
    tool_names = {tool["name"] for tool in get_tools()}

    optional_tools = {
        "run_pseudobulk_deg": "scagent.analysis.pseudobulk",
        "run_spectra": "scagent.analysis.spectra",
    }
    for tool_name, module_name in optional_tools.items():
        assert (tool_name in tool_names) == (importlib.util.find_spec(module_name) is not None)


def test_data_input_discovery_tool_is_advertised():
    tool_names = {tool["name"] for tool in get_tools()}

    assert "inspect_data_inputs" in tool_names


def test_concat_datasets_preserves_explicit_sample_labels_and_counts():
    first = _dataset("a", 2)
    second = _dataset("b", 3)

    combined = concat_datasets(
        [first, second],
        batch_key="replicate",
        batch_names=["Rep1", "Rep2"],
    )

    assert combined.obs["replicate"].value_counts().to_dict() == {
        "Rep2": 3,
        "Rep1": 2,
    }
    assert combined.var_names.is_unique
    assert combined.obs_names.is_unique
    assert combined.var["feature_type"].eq("Gene Expression").all()


def test_concat_datasets_rejects_ambiguous_sample_names():
    datasets = [_dataset("a", 1), _dataset("b", 1)]

    try:
        concat_datasets(datasets, batch_names=["same", "same"])
    except ValueError as exc:
        assert "must be unique" in str(exc)
    else:
        raise AssertionError("duplicate sample names should fail")


def test_concat_datasets_honors_outer_and_inner_join():
    first = ad.AnnData(
        np.ones((1, 2)),
        obs=pd.DataFrame(index=["a"]),
        var=pd.DataFrame(index=["GENE1", "GENE2"]),
    )
    second = ad.AnnData(
        np.ones((1, 2)),
        obs=pd.DataFrame(index=["b"]),
        var=pd.DataFrame(index=["GENE2", "GENE3"]),
    )

    outer = concat_datasets(
        [first.copy(), second.copy()],
        batch_names=["Rep1", "Rep2"],
        join="outer",
    )
    inner = concat_datasets(
        [first.copy(), second.copy()],
        batch_names=["Rep1", "Rep2"],
        join="inner",
    )

    assert list(outer.var_names) == ["GENE1", "GENE2", "GENE3"]
    assert list(inner.var_names) == ["GENE2"]


def test_discover_data_inputs_separates_source_files_from_combined_outputs(tmp_path):
    (tmp_path / "Rep1.h5").touch()
    (tmp_path / "Rep2.h5").touch()
    (tmp_path / "combined_replicates.h5ad").touch()
    (tmp_path / "notes.txt").touch()

    result = discover_data_inputs(tmp_path)

    assert [item["name"] for item in result["source_datasets"]] == [
        "Rep1.h5",
        "Rep2.h5",
    ]
    assert [item["name"] for item in result["likely_combined_outputs"]] == [
        "combined_replicates.h5ad"
    ]
