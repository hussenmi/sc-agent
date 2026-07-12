"""Stub for tiledb.vector_search on aarch64 (no upstream build).

scimilarity.cell_search_knn.load_knn_index does `import tiledb.vector_search as vs`
unconditionally, but only calls vs.IVFFlatIndex for knn_type='tiledb_vector_search'.
The default annotation path (knn_type='hnswlib') imports this module and never
touches it. Any real use raises a clear error."""


class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError(
            "tiledb.vector_search is stubbed on this aarch64 host (no upstream build). "
            "Only knn_type='hnswlib' works here; the tiledb-vector-search IVFFlatIndex / "
            "CellQuery path is unavailable.")


class IVFFlatIndex(_Unavailable):
    pass
