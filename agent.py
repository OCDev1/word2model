"""Orchestrator: description -> LLM -> CadQuery (CQGI) -> STL, with retry on failure."""

from pathlib import Path

from . import cadquery_runner
from . import llm


def description_to_stl(
    description: str,
    output_path: str | Path,
    *,
    max_retries: int = 2,
    provider: str = "openai",
    model: str | None = None,
) -> tuple[list[Path], str | None]:
    """
    Generate a 3D model from a natural language description and export to STL.

    Returns (list of written STL paths, error message if failed).
    On success, error is None. On failure after all retries, paths may be empty and error is set.
    provider: openai | anthropic | google (or gemini). model: optional override per provider.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            code = llm.generate_cadquery_code(
                description,
                previous_error=last_error,
                provider=provider,
                model=model,
            )
        except Exception as e:
            return [], str(e)

        build_result = cadquery_runner.run_script(code)

        if build_result.success:
            written = cadquery_runner.export_build_result_to_stl(build_result, output_path)
            return written, None

        exc = build_result.exception
        last_error = f"{type(exc).__name__}: {exc}"
        if getattr(exc, "__traceback__", None):
            import traceback
            last_error += "\n" + "".join(traceback.format_tb(exc.__traceback__))

    return [], f"Failed after {max_retries + 1} attempt(s). Last error: {last_error}"
