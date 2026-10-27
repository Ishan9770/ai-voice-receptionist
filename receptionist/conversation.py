"""Conversation memory. Keeps the last N turns; state changes only on success."""
from __future__ import annotations

from .llm import LLMProvider


class Conversation:
    def __init__(self, system_prompt: str, max_turns: int = 12):
        self._system = {"role": "system", "content": system_prompt}
        self._history: list[dict] = []
        self._max_messages = max_turns * 2

    def respond(self, user_text: str, llm: LLMProvider) -> str:
        messages = [self._system, *self._history, {"role": "user", "content": user_text}]
        reply = llm.generate(messages)  # may raise LLMError: history stays untouched
        self._history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        self._history = self._history[-self._max_messages:]
        return reply

    @property
    def turns(self) -> int:
        return len(self._history) // 2
