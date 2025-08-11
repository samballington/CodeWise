#!/usr/bin/env python3
"""
JSON Prompt schema models and validation utilities.

Implements the project-agnostic structured JSON response described in
JSON_PROMPT_IMPLEMENTATION.md:

{
  "response": {
    "metadata": { ... },
    "sections": [ { ...discriminated by type... } ],
    "follow_up_suggestions": [ ... ]
  }
}
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError


class ResponseMetadata(BaseModel):
    query_type: Literal["architecture", "dependencies", "database", "general"] = "general"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    execution_time: Optional[float] = Field(default=None, ge=0.0)
    tools_used: Optional[List[str]] = None


class SectionHeading(BaseModel):
    type: Literal["heading"]
    level: int = Field(ge=1, le=6)
    content: str


class SectionParagraph(BaseModel):
    type: Literal["paragraph"]
    content: str


class SectionTable(BaseModel):
    type: Literal["table"]
    title: Optional[str] = None
    columns: List[str]
    rows: List[List[Union[str, int, float, None]]]
    note: Optional[str] = None


class SectionList(BaseModel):
    type: Literal["list"]
    style: Literal["bullet", "numbered"] = "bullet"
    title: Optional[str] = None
    items: List[str]


class SectionCodeBlock(BaseModel):
    type: Literal["code_block"]
    language: Optional[str] = None
    title: Optional[str] = None
    content: str


class SectionCallout(BaseModel):
    type: Literal["callout"]
    style: Literal["info", "warning", "error", "success"] = "info"
    title: Optional[str] = None
    content: str


class TreeNode(BaseModel):
    label: str
    children: Optional[List["TreeNode"]] = None


TreeNode.update_forward_refs()


class SectionTree(BaseModel):
    type: Literal["tree"]
    title: Optional[str] = None
    root: TreeNode


class SectionQuote(BaseModel):
    type: Literal["quote"]
    content: str
    attribution: Optional[str] = None


class SectionDivider(BaseModel):
    type: Literal["divider"]


class SectionImage(BaseModel):
    type: Literal["image"]
    src: str
    alt: Optional[str] = None
    caption: Optional[str] = None

class SectionDiagram(BaseModel):
    type: Literal["diagram"]
    format: Literal["mermaid"] = "mermaid"
    title: Optional[str] = None
    content: str  # raw diagram DSL, no code fences


Section = Union[
    SectionHeading,
    SectionParagraph,
    SectionTable,
    SectionList,
    SectionCodeBlock,
    SectionCallout,
    SectionTree,
    SectionQuote,
    SectionDivider,
    SectionImage,
    SectionDiagram,
]


class JsonPromptResponseBody(BaseModel):
    metadata: ResponseMetadata
    sections: List[Section]
    follow_up_suggestions: List[str] = Field(default_factory=list)


class JsonPromptEnvelope(BaseModel):
    response: JsonPromptResponseBody


def parse_json_prompt(text: str) -> Optional[JsonPromptEnvelope]:
    """Attempt to parse a JSON Prompt response from a raw string.

    Returns the parsed model on success, or None if parsing/validation fails.
    """
    import json

    # Fast path: if the string contains a fenced JSON block, try the last one
    candidate = text.strip()
    try:
        # If it's a fenced block
        if candidate.startswith("```"):
            # Extract last fenced json block
            import re
            blocks = re.findall(r"```json[\s\S]*?```", candidate)
            if blocks:
                candidate = blocks[-1].removeprefix("```json").removesuffix("```").strip()
                data = json.loads(candidate)
                return JsonPromptEnvelope.model_validate(data)
        
        # Try direct JSON parse of the full string first
        try:
            data = json.loads(candidate)
            return JsonPromptEnvelope.model_validate(data)
        except json.JSONDecodeError:
            pass

        # Balanced-brace extraction: find the last valid top-level JSON object in text
        start_indices = [i for i, ch in enumerate(candidate) if ch == '{']
        for start in reversed(start_indices):
            depth = 0
            for end in range(start, len(candidate)):
                ch = candidate[end]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        snippet = candidate[start:end + 1]
                        try:
                            data = json.loads(snippet)
                            return JsonPromptEnvelope.model_validate(data)
                        except (json.JSONDecodeError, ValidationError):
                            # One targeted repair: if failure seems due to unescaped quotes
                            # inside a diagram section's content, attempt a minimal fix by
                            # escaping unescaped quotes in that specific string value and retry.
                            try:
                                import re
                                # Find "type":"diagram" blocks and their content fields
                                def repair_diagram_quotes(s: str) -> str:
                                    # Replace occurrences like ["Label with \" quotes"] safely
                                    def _fix_content(m: re.Match) -> str:
                                        prefix = m.group(1)  # up to opening quote of content
                                        body = m.group(2)
                                        # Escape " that are not already escaped
                                        body_fixed = re.sub(r'(?<!\\)"', r'\\"', body)
                                        return f"{prefix}{body_fixed}\""

                                    # Match the content string value, capture its raw body (group 2)
                                    pattern = r'(\"type\"\s*:\s*\"diagram\"[\s\S]*?\"content\"\s*:\s*\")(.*?)(\")'
                                    return re.sub(pattern, lambda mm: _fix_content(mm), s)

                                repaired = repair_diagram_quotes(snippet)
                                if repaired != snippet:
                                    data2 = json.loads(repaired)
                                    return JsonPromptEnvelope.model_validate(data2)
                            except Exception:
                                pass
                            break  # move to previous start
        return None
    except (ValidationError, Exception):
        return None


__all__ = [
    "ResponseMetadata",
    "JsonPromptResponseBody",
    "JsonPromptEnvelope",
    "Section",
    "parse_json_prompt",
]

