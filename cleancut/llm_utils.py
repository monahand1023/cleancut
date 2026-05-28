"""Shared helpers for LLM/VLM classifiers."""

from __future__ import annotations

import re


def strip_to_json(text: str) -> str:
    """Extract the first JSON object from a possibly chatty LLM response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def make_ollama_client(host: str | None):
    """Create an Ollama client, optionally with a custom host."""
    import ollama
    return ollama.Client(host=host) if host else ollama.Client()
