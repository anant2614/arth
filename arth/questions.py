"""Internal question representation shared by students, teacher and API."""

from __future__ import annotations

from dataclasses import dataclass, field

from .concepts import BUILTIN_BOOLS, BUILTIN_TAXONOMIES, BuiltinBool, BuiltinTaxonomy

MAX_OPTIONS = 255


@dataclass(frozen=True)
class Question:
    name: str
    type: str  # "choice" | "bool"
    options: tuple[tuple[str, str], ...] = field(default=())  # (option, description)
    description: str | None = None

    @property
    def option_names(self) -> list[str]:
        return [o for o, _ in self.options]

    def option_texts(self) -> list[str]:
        """Text embedded for each option: its name in words plus its description."""
        return [f"{o.replace('_', ' ')}: {d}" if d else o.replace("_", " ") for o, d in self.options]

    def bool_text(self) -> str:
        return (self.description or self.name.removeprefix("is_").replace("_", " ")).strip()

    def signature(self) -> tuple:
        return (self.type, self.options if self.type == "choice" else self.bool_text())

    @classmethod
    def choice(cls, name: str, options: dict[str, str]) -> "Question":
        return cls(name, "choice", tuple(options.items()))

    @classmethod
    def boolean(cls, name: str, description: str | None = None) -> "Question":
        return cls(name, "bool", (), description)

    @classmethod
    def from_builtin(cls, b: BuiltinTaxonomy | BuiltinBool, name: str | None = None) -> "Question":
        if isinstance(b, BuiltinTaxonomy):
            return cls.choice(name or b.name, b.options)
        return cls.boolean(name or b.name, b.description)


def builtin_question(name: str) -> Question:
    if name in BUILTIN_TAXONOMIES:
        return Question.from_builtin(BUILTIN_TAXONOMIES[name])
    return Question.from_builtin(BUILTIN_BOOLS[name])


def concept_label(q_source: BuiltinTaxonomy | BuiltinBool, concept: str):
    return q_source.label(concept)
