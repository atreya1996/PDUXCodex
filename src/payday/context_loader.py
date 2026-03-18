"""Utilities for loading durable product context into reusable prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ContextMemory:
    """Loaded markdown memory used to build analysis and classification prompts."""

    product: str
    research_rules: str
    analysis_prompt: str

    def as_mapping(self) -> dict[str, str]:
        """Return the loaded context as a plain dictionary."""
        return {
            "product": self.product,
            "research_rules": self.research_rules,
            "analysis_prompt": self.analysis_prompt,
        }


DEFAULT_CONTEXT_FILES: dict[str, str] = {
    "product": "context/product.md",
    "research_rules": "context/research_rules.md",
    "analysis_prompt": "prompts/analysis_prompt.md",
}

JSON_ONLY_INSTRUCTION = (
    "Return structured JSON only. Do not include markdown fences or explanatory text."
)


def repo_root() -> Path:
    """Infer the repository root from this file location."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_root(root: Path | str | None = None) -> Path:
    """Resolve an explicit repository root or fall back to the inferred root."""
    return Path(root).resolve() if root is not None else repo_root()


def read_context_file(relative_path: str, root: Path | str | None = None) -> str:
    """Read a context file relative to the repository root."""
    path = resolve_repo_root(root) / relative_path
    return path.read_text(encoding="utf-8").strip()


def load_context_memory(
    root: Path | str | None = None,
    file_map: Mapping[str, str] | None = None,
) -> ContextMemory:
    """Load the durable context bundle from disk."""
    files = dict(DEFAULT_CONTEXT_FILES)
    if file_map:
        files.update(file_map)

    return ContextMemory(
        product=read_context_file(files["product"], root=root),
        research_rules=read_context_file(files["research_rules"], root=root),
        analysis_prompt=read_context_file(files["analysis_prompt"], root=root),
    )


def build_memory_block(memory: ContextMemory | None = None) -> str:
    """Render the markdown memory as a prompt-ready block."""
    loaded = memory or load_context_memory()
    return "\n\n".join(
        [
            "[Product Context]\n" + loaded.product,
            "[Research Rules]\n" + loaded.research_rules,
            "[Analysis Prompt Memory]\n" + loaded.analysis_prompt,
            "[Output Constraint]\n" + JSON_ONLY_INSTRUCTION,
        ]
    )


def inject_context(
    prompt: str,
    *,
    memory: ContextMemory | None = None,
    heading: str = "Use the following durable context before completing the task.",
) -> str:
    """Append durable context memory to an arbitrary prompt."""
    prompt = prompt.strip()
    return f"{prompt}\n\n{heading}\n\n{build_memory_block(memory)}".strip()


def build_analysis_prompt(
    task: str,
    *,
    memory: ContextMemory | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> str:
    """Build a reusable analysis prompt with durable memory injected."""
    sections = [
        "You are performing product analysis for PayDay.",
        task.strip(),
    ]
    if extra_instructions:
        sections.append(
            "Additional instructions:\n- " + "\n- ".join(x.strip() for x in extra_instructions)
        )
    sections.append(JSON_ONLY_INSTRUCTION)
    return inject_context("\n\n".join(sections), memory=memory)


def build_classification_prompt(
    item_to_classify: str,
    *,
    classes: Iterable[str] | None = None,
    memory: ContextMemory | None = None,
    extra_instructions: Iterable[str] | None = None,
) -> str:
    """Build a reusable classification prompt with durable memory injected."""
    sections = [
        "You are performing user or research classification for PayDay.",
        f"Item to classify:\n{item_to_classify.strip()}",
    ]
    if classes:
        sections.append("Allowed classes:\n- " + "\n- ".join(x.strip() for x in classes))
    if extra_instructions:
        sections.append(
            "Additional instructions:\n- " + "\n- ".join(x.strip() for x in extra_instructions)
        )
    sections.append(JSON_ONLY_INSTRUCTION)
    return inject_context("\n\n".join(sections), memory=memory)
