"""Provider-agnostic LLM layer for generating CadQuery code. Supports OpenAI, Anthropic, Google Gemini."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

from . import prompts

# Load .env when this module is used (e.g. by web server) so API keys are available
load_dotenv(Path(__file__).resolve().parent / ".env")

# Type for a completion callable: (system, user) -> response text
CompleteFn = Callable[[str, str], str]


class LLMProvider(Protocol):
    """Protocol for an LLM that can complete a chat turn given system and user text."""

    def complete(self, system: str, user: str) -> str: ...


def _strip_code_block(text: str) -> str:
    """Remove optional markdown code fence so we get plain Python."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# --- OpenAI ---

def _openai_complete(system: str, user: str, *, model: str, api_key: str | None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


# --- Anthropic ---

def _anthropic_complete(system: str, user: str, *, model: str, api_key: str | None) -> str:
    from anthropic import Anthropic
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if not message.content or message.content[0].type != "text":
        return ""
    return message.content[0].text.strip()


# --- Google Gemini ---

def _gemini_complete(system: str, user: str, *, model: str, api_key: str | None) -> str:
    from google import genai
    from google.genai.types import GenerateContentConfig
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
        ),
    )
    if not response.text:
        return ""
    return response.text.strip()


# Registry: provider name -> (default_model, complete_fn)
_PROVIDERS: dict[str, tuple[str, Callable[..., str]]] = {
    "openai": ("gpt-4o-mini", _openai_complete),
    "anthropic": ("claude-3-5-haiku-20241022", _anthropic_complete),
    "google": ("gemini-2.0-flash", _gemini_complete),
    "gemini": ("gemini-2.0-flash", _gemini_complete),
}


def get_provider(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
) -> CompleteFn:
    """
    Return a completion function for the given provider.
    provider: one of "openai", "anthropic", "google" (or "gemini").
    model: override default model for that provider.
    api_key: override env var (optional).
    """
    key = provider.lower().strip()
    if key not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {', '.join(_PROVIDERS)}"
        )
    default_model, complete_fn = _PROVIDERS[key]
    chosen_model = model or default_model

    def complete(system: str, user: str) -> str:
        return complete_fn(system, user, model=chosen_model, api_key=api_key)

    return complete


def list_providers() -> list[str]:
    """Return registered provider names (excluding aliases)."""
    return ["openai", "anthropic", "google"]


def get_default_model(provider: str) -> str:
    """Return the default model for a provider."""
    key = provider.lower().strip()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    return _PROVIDERS[key][0]


def generate_cadquery_code(
    description: str,
    *,
    previous_error: str | None = None,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Ask the LLM for CadQuery (CQGI) code matching the description.
    If previous_error is set, the prompt includes it for a retry.
    provider: openai | anthropic | google (or gemini).
    """
    complete = get_provider(provider, model=model, api_key=api_key)

    if previous_error:
        user_content = prompts.USER_PROMPT_WITH_ERROR.format(
            description=description,
            error=previous_error,
        )
    else:
        user_content = prompts.USER_PROMPT.format(description=description)

    raw = complete(prompts.SYSTEM_PROMPT, user_content)
    return _strip_code_block(raw)
