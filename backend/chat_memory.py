import os
from typing import List, Tuple
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatMemory:
    """Rolling chat history with automatic summarisation to keep prompts short."""

    def __init__(self, max_recent: int = 10, summary_max_tokens: int = 300):
        self.max_recent = max_recent
        self.summary_max_tokens = summary_max_tokens
        self.recent: List[Tuple[str, str]] = []  # (role, content)
        self.summary: str = ""  # running summary of older messages

    def add_message(self, role: str, content: str):
        """Add a user or assistant message and summarise if needed."""
        if role not in {"user", "assistant"}:
            return
        self.recent.append((role, content))
        if len(self.recent) > self.max_recent:
            # Pop oldest message(s) and fold into summary
            old = self.recent.pop(0)
            self._update_summary([old])

    def _update_summary(self, messages: List[Tuple[str, str]]):
        """Summarise given messages together with existing summary using cheap model."""
        try:
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a summarization assistant. You receive pieces of a chat; "
                        "produce a concise running summary that preserves key technical details. "
                        "Keep it under {self.summary_max_tokens} tokens."
                    ),
                }
            ]
            if self.summary:
                prompt_messages.append(
                    {"role": "user", "content": f"Current summary: {self.summary}"}
                )
            joined = "\n".join(f"{r.capitalize()}: {c}" for r, c in messages)
            prompt_messages.append(
                {"role": "user", "content": f"New dialogue to add: {joined}"}
            )

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=prompt_messages,
                max_tokens= self.summary_max_tokens,
                temperature=0,
            )
            self.summary = response.choices[0].message.content.strip()
        except Exception:
            # Fallback: append raw text truncated
            raw = " ".join(c for _, c in messages)
            self.summary = (self.summary + " " + raw)[-4000:]

    def as_langchain_messages(self):
        """Return list of langchain BaseMessages representing the history."""
        from langchain.schema import SystemMessage, HumanMessage, AIMessage

        msgs: List = []
        if self.summary:
            msgs.append(SystemMessage(content=f"Conversation summary: {self.summary}"))
        for role, content in self.recent:
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))
        return msgs 