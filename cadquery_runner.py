"""CQGI wrapper: run CadQuery script and export results to STL."""

from pathlib import Path

import cadquery as cq
import cadquery.cqgi as cqgi


def run_script(script: str) -> cqgi.BuildResult:
    """Parse and execute a CadQuery (CQGI) script. Returns the BuildResult."""
    model = cqgi.parse(script)
    return model.build()


def export_build_result_to_stl(
    build_result: cqgi.BuildResult,
    output_path: str | Path,
) -> list[Path]:
    """
    Export all shapes from a successful BuildResult to STL files.
    If there is one result, writes to output_path as-is.
    If there are multiple results, writes output_0.stl, output_1.stl, ... (same stem as output_path).
    Returns the list of written file paths.
    """
    if not build_result.success:
        raise ValueError("Cannot export failed build result", build_result.exception)

    results = build_result.results
    if not results:
        raise ValueError("Build succeeded but no shapes returned (script must call show_object)")

    output_path = Path(output_path)
    stem = output_path.stem
    parent = output_path.parent
    written: list[Path] = []

    if len(results) == 1:
        path = parent / f"{stem}.stl"
        cq.exporters.export(results[0].shape, str(path))
        written.append(path)
    else:
        for i, result in enumerate(results):
            path = parent / f"{stem}_{i}.stl"
            cq.exporters.export(result.shape, str(path))
            written.append(path)

    return written
