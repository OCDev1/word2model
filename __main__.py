"""CLI: description -> STL via Text-to-3D agent."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import agent
from . import llm

# Load .env from project root (directory containing this package)
load_dotenv(Path(__file__).resolve().parent / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a printable .stl from a natural language description using CadQuery.",
    )
    parser.add_argument(
        "description",
        type=str,
        help="Natural language description of the 3D model (e.g. 'a 20mm cube with a 5mm hole through the center').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.stl",
        help="Output path for the .stl file (default: output.stl). Multiple files get suffixes _0, _1 if the model has multiple parts.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        metavar="N",
        help="Max number of retries with error feedback when generated code fails (default: 2).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=llm.list_providers() + ["gemini"],
        help="LLM provider: openai, anthropic, google, or gemini (alias for google). Default: openai.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional). Uses provider default if not set, e.g. gpt-4o-mini, claude-3-5-haiku, gemini-2.0-flash.",
    )
    args = parser.parse_args()

    paths, err = agent.description_to_stl(
        args.description,
        args.output,
        max_retries=args.retries,
        provider=args.provider,
        model=args.model,
    )

    if err:
        print("Error:", err, file=sys.stderr)
        sys.exit(1)
    for p in paths:
        print("Wrote", p)


if __name__ == "__main__":
    main()
