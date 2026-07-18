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


def _write_matrix_csv(path, n_cells, sep=","):
    header = sep.join(["gene"] + [f"cell{i}" for i in range(n_cells)])
    row = sep.join(["GENE1"] + ["1"] * n_cells)
    path.write_text(header + "\n" + row + "\n")


def test_discover_data_inputs_recognizes_csv_count_matrices(tmp_path):
    # The run_2026_07_15_121111 shape: a folder of per-sample CSV count matrices.
    # These were previously invisible to discovery (only h5ad/h5/mtx were), so the
    # multi_dataset_loading checkpoint never fired for them.
    for i in range(4):
        _write_matrix_csv(tmp_path / f"GSM55734{i:02d}_sample{i}.csv", n_cells=8)
    result = discover_data_inputs(tmp_path)
    assert result["n_source_datasets"] == 4
    assert all(d["format"] == "csv" for d in result["source_datasets"])
    assert all(d["delimiter"] == "," for d in result["source_datasets"])


def test_discover_data_inputs_recognizes_gzipped_tsv(tmp_path):
    import gzip

    for i in range(2):
        with gzip.open(tmp_path / f"s{i}.tsv.gz", "wt") as handle:
            handle.write("gene\t" + "\t".join(f"c{j}" for j in range(6)) + "\n")
            handle.write("G1\t" + "\t".join("1" for _ in range(6)) + "\n")
    result = discover_data_inputs(tmp_path)
    assert result["n_source_datasets"] == 2
    assert result["source_datasets"][0]["delimiter"] == "\t"


def test_discover_data_inputs_excludes_stray_metadata_table(tmp_path):
    # A metadata table beside real matrices must not join the replicate group: it
    # has a different column count, so structural grouping sets it aside.
    for i in range(3):
        _write_matrix_csv(tmp_path / f"sample{i}.csv", n_cells=40)
    (tmp_path / "metadata.csv").write_text("sample_id,condition,age\nsample0,tumor,55\n")
    result = discover_data_inputs(tmp_path)
    assert sorted(d["name"] for d in result["source_datasets"]) == [
        "sample0.csv",
        "sample1.csv",
        "sample2.csv",
    ]
    assert [d["name"] for d in result["excluded_datasets"]] == ["metadata.csv"]


def test_discover_data_inputs_single_csv_is_one_source(tmp_path):
    _write_matrix_csv(tmp_path / "only.csv", n_cells=10)
    result = discover_data_inputs(tmp_path / "only.csv")
    assert result["n_source_datasets"] == 1


def test_discover_data_inputs_treats_10x_directory_as_single_dataset(tmp_path):
    # A 10x bundle's loose barcodes/features tsv files must not be counted as
    # separate per-sample tables now that tsv is recognized.
    (tmp_path / "matrix.mtx").write_text("%%MatrixMarket\n1 1 1\n1 1 1\n")
    (tmp_path / "features.tsv").write_text("ENSG1\tGENE1\tGene Expression\n")
    (tmp_path / "barcodes.tsv").write_text("AAAA-1\n")
    result = discover_data_inputs(tmp_path)
    assert result["n_source_datasets"] == 1
    assert result["source_datasets"][0]["format"] == "10x_mtx_directory"
