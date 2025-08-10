#!/usr/bin/env python3
"""Lightweight post-processing to improve readability of JSON Prompt sections.

Rules (see docs/paragraph-formatting-plan.md):
- Convert inline enumerations inside paragraphs into a list section
- Remove markdown emphasis/inline code from paragraphs
- Split very long paragraphs into smaller ones
- Optional: Add Mermaid flow diagram if many arrows are present
"""

from __future__ import annotations

from typing import Dict, Any, List
import re
from mermaid_utils import validate_mermaid_simple
import logging

logger = logging.getLogger(__name__)


def _strip_markdown_inline(text: str) -> str:
    # Remove **bold** and inline `code`
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def _paragraph_to_list_if_enumerated(content: str) -> List[Dict[str, Any]] | None:
    # Look for patterns like: sentence - item - item - item
    if " - " not in content:
        return None
    # Heuristic: treat as list if there are at least two bullet-like splits
    parts = [p.strip(" -\n\t") for p in content.split(" - ")]
    if len(parts) < 3:
        return None
    # The first chunk might contain an intro sentence; keep it as paragraph if substantial
    intro = parts[0].strip()
    items = [p for p in parts[1:] if p]
    out: List[Dict[str, Any]] = []
    if intro and len(intro) > 30:
        out.append({"type": "paragraph", "content": intro})
    if items:
        out.append({"type": "list", "style": "bullet", "items": items})
    return out or None


def _split_long_paragraph(content: str, max_len: int = 600) -> List[str]:
    if len(content) <= max_len:
        return [content]
    # Split by period+space heuristics
    sentences = re.split(r"(?<=\.)\s+", content)
    chunks: List[str] = []
    acc = ""
    for s in sentences:
        candidate = (acc + " " + s).strip() if acc else s
        if len(candidate) <= max_len or not acc:
            acc = candidate
        else:
            chunks.append(acc)
            acc = s
        if len(chunks) >= 2:  # limit to 3 chunks total
            break
    if acc:
        chunks.append(acc)
    return chunks


def _validate_and_fix_mermaid_diagram(diagram_section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix Mermaid diagram content using the validation system
    """
    try:
        original_content = diagram_section.get("content", "")
        if not original_content or not original_content.strip():
            return diagram_section
        
        # Validate and correct the Mermaid code
        corrected_content, is_valid = validate_mermaid_simple(original_content.strip())
        
        if is_valid and corrected_content != original_content:
            logger.info(f"✅ Mermaid diagram corrected and validated successfully")
            # Return corrected diagram
            corrected_section = diagram_section.copy()
            corrected_section["content"] = corrected_content
            return corrected_section
        elif is_valid:
            logger.debug("✅ Mermaid diagram validated successfully (no corrections needed)")
            return diagram_section
        else:
            logger.warning(f"❌ Mermaid diagram validation failed, keeping original")
            # Keep original content but maybe add a note about validation failure
            return diagram_section
            
    except Exception as e:
        logger.warning(f"Mermaid validation error: {e}, keeping original diagram")
        return diagram_section


def _maybe_mermaid_from_flow(content: str) -> Dict[str, Any] | None:
    # Basic detection of arrow-heavy flows
    if content.count("→") < 2:
        return None
    # Extract nodes by splitting on arrows, sanitize labels
    nodes = [re.sub(r"\s+", " ", n.strip()) for n in content.split("→") if n.strip()]
    if len(nodes) < 3:
        return None
    edges = []
    for a, b in zip(nodes, nodes[1:]):
        edges.append(f"  {a.replace(' ', '_')}-->{b.replace(' ', '_')}")
    mermaid_content = "graph LR\n" + "\n".join(edges)
    
    # Validate the generated Mermaid content
    try:
        corrected_content, is_valid = validate_mermaid_simple(mermaid_content)
        if is_valid:
            return {"type": "diagram", "format": "mermaid", "title": "Flow", "content": corrected_content}
    except Exception:
        pass
    
    # Fallback to code block if validation fails
    return {"type": "code_block", "language": "mermaid", "title": "Flow", "content": mermaid_content}


def improve_json_prompt_readability(envelope: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = envelope.get("response") or {}
        sections = list(response.get("sections") or [])
        improved: List[Dict[str, Any]] = []
        for sec in sections:
            t = sec.get("type")
            if t == "paragraph":
                content = str(sec.get("content", ""))
                content = _strip_markdown_inline(content)

                # Paragraph → list if enumerations detected
                converted = _paragraph_to_list_if_enumerated(content)
                if converted:
                    improved.extend(converted)
                    # Optional mermaid
                    m = _maybe_mermaid_from_flow(content)
                    if m:
                        improved.append(m)
                    continue

                # Split long paragraphs
                parts = _split_long_paragraph(content)
                for p in parts:
                    improved.append({"type": "paragraph", "content": p})
                # Optional mermaid
                m = _maybe_mermaid_from_flow(content)
                if m:
                    improved.append(m)
                continue

            # Validate and fix Mermaid diagrams
            elif t == "diagram" and sec.get("format") == "mermaid":
                validated_section = _validate_and_fix_mermaid_diagram(sec)
                improved.append(validated_section)
                continue
            
            # passthrough for other section types
            improved.append(sec)

        envelope = dict(envelope)
        envelope.setdefault("response", {}).update({"sections": improved})
        return envelope
    except Exception:
        return envelope


__all__ = ["improve_json_prompt_readability"]




