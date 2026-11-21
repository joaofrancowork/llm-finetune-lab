"""Prompts for the two settings being compared.

* Zero-shot: the base model must pick from all 77 labels listed in the prompt
  (430 prompt tokens per request with Qwen's chat template).
* Fine-tuned: the adapter has learned the label set, so the prompt is one line.
  Same output format in both: the label alone, in snake_case.
"""

from __future__ import annotations

from typing import Protocol

FINETUNED_SYSTEM = (
    "Classify the banking customer message into its intent. Reply with the intent label only."
)


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self, conversation: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


def zero_shot_system(labels: list[str]) -> str:
    return (
        "You classify messages sent to a bank's customer support into exactly one intent.\n"
        "Reply with one label from the list, copied exactly, and nothing else.\n\n"
        "Labels:\n" + "\n".join(labels)
    )


def messages(text: str, system: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": text.strip()}]


def render(tokenizer: ChatTokenizer, text: str, system: str) -> str:
    """The prompt string the model sees, ending where the answer should start."""
    return tokenizer.apply_chat_template(
        messages(text, system), tokenize=False, add_generation_prompt=True
    )
