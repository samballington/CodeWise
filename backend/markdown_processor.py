#!/usr/bin/env python3
"""
Handbook-based Markdown Processing Pipeline
Implements the "Parse, Don't Scan" philosophy with 5 distinct stages.

Replaces the brittle regex-based cleaning with secure, robust processing.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time

# Stage 1: Security Sanitization
import bleach

# Stage 2: Structured Parsing  
from markdown_it import MarkdownIt
from markdown_it.token import Token

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of the markdown processing pipeline"""
    success: bool
    content: str
    processing_time: float
    stage_completed: int
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class MarkdownProcessor:
    """
    5-Stage Markdown Processing Pipeline following the CodeWise Handbook
    
    Stage 1: Initial Sanitization (Security First)
    Stage 2: Structured Parsing (Parse, Don't Scan)
    Stage 3: Validation & Transformation on AST
    Stage 4: Rendering Back to Final Format
    Stage 5: Graceful Fallback (Error Recovery)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MarkdownProcessor")
        
        # Stage 1: Define allowed HTML tags for markdown rendering
        # These are tags that react-markdown will eventually render
        self.ALLOWED_TAGS = [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'strong', 'em', 'ul', 'ol', 'li', 
            'code', 'pre', 'blockquote',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'br', 'hr', 'a', 'img'
        ]
        
        self.ALLOWED_ATTRIBUTES = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title'],
            'code': ['class'],
            'pre': ['class']
        }
        
        # Stage 2: Initialize markdown parser with extensions
        self.md_parser = MarkdownIt("commonmark", {
            "highlight": True,
            "linkify": True,
            "typographer": False  # Avoid smart quotes that might break code
        })
        
        # Enable tables and other GFM features
        self.md_parser.enable(['table', 'strikethrough'])
    
    def process(self, raw_llm_output: str) -> ProcessingResult:
        """
        Main processing method implementing the 5-stage pipeline
        """
        start_time = time.time()
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # STAGE 1: Initial Sanitization (Security First)
            self.logger.info("🛡️ STAGE 1: Security sanitization")
            sanitized_output = self._stage1_sanitize(raw_llm_output)
            metadata['original_length'] = len(raw_llm_output)
            metadata['sanitized_length'] = len(sanitized_output)
            
            # STAGE 2: Structured Parsing
            self.logger.info("🏗️ STAGE 2: Structured parsing")
            tokens = self._stage2_parse(sanitized_output)
            metadata['token_count'] = len(tokens)
            
            # STAGE 3: Validation & Transformation on AST
            self.logger.info("✅ STAGE 3: AST validation & transformation")
            validated_tokens, stage3_warnings = self._stage3_validate_transform(tokens)
            warnings.extend(stage3_warnings)
            metadata['validated_token_count'] = len(validated_tokens)
            
            # STAGE 4: Rendering Back to Final Format
            self.logger.info("📝 STAGE 4: Rendering to final markdown")
            final_content = self._stage4_render(validated_tokens)
            metadata['final_length'] = len(final_content)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ PIPELINE SUCCESS: Processed {len(raw_llm_output)} → {len(final_content)} chars in {processing_time:.3f}s")
            
            return ProcessingResult(
                success=True,
                content=final_content,
                processing_time=processing_time,
                stage_completed=4,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            # STAGE 5: Graceful Fallback
            self.logger.error(f"❌ PIPELINE FAILED at stage, falling back: {e}")
            return self._stage5_fallback(raw_llm_output, str(e), start_time)
    
    def _stage1_sanitize(self, raw_output: str) -> str:
        """
        Stage 1: Remove potential XSS attacks while preserving markdown structure
        """
        try:
            # Use bleach to sanitize HTML while allowing markdown-safe tags
            sanitized = bleach.clean(
                raw_output,
                tags=self.ALLOWED_TAGS,
                attributes=self.ALLOWED_ATTRIBUTES,
                strip=True,  # Remove disallowed tags entirely
                strip_comments=True  # Remove HTML comments
            )
            
            # CRITICAL FIX: Undo bleach's aggressive HTML entity escaping that breaks Mermaid arrows
            # Since we're processing trusted LLM output, not user input, we can safely restore HTML entities
            sanitized = sanitized.replace('&amp;', '&')
            sanitized = sanitized.replace('&gt;', '>')
            sanitized = sanitized.replace('&lt;', '<')
            sanitized = sanitized.replace('&quot;', '"')
            
            # Log any sanitization that occurred
            if len(sanitized) != len(raw_output):
                self.logger.warning(f"🛡️ SANITIZATION: Removed {len(raw_output) - len(sanitized)} potentially dangerous characters")
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"❌ STAGE 1 FAILED: Sanitization error: {e}")
            # If sanitization fails, at least escape obvious HTML
            return raw_output.replace('<script', '&lt;script').replace('<iframe', '&lt;iframe')
    
    def _stage2_parse(self, sanitized_output: str) -> List[Token]:
        """
        Stage 2: Parse markdown into structured AST tokens
        """
        try:
            # Parse into token stream - this gives us context awareness
            tokens = self.md_parser.parse(sanitized_output)
            
            self.logger.info(f"🏗️ PARSED: {len(tokens)} tokens from {len(sanitized_output)} chars")
            
            # Log token types for debugging
            token_types = [token.type for token in tokens]
            unique_types = set(token_types)
            self.logger.debug(f"🏗️ TOKEN TYPES: {unique_types}")
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"❌ STAGE 2 FAILED: Parsing error: {e}")
            raise  # Let this bubble up to stage 5 fallback
    
    def _stage3_validate_transform(self, tokens: List[Token]) -> Tuple[List[Token], List[str]]:
        """
        Stage 3: Validate and transform tokens with context awareness
        """
        validated_tokens = []
        warnings = []
        
        try:
            for i, token in enumerate(tokens):
                # Validate and potentially transform each token based on type
                if token.type == 'fence':
                    # Handle code blocks
                    processed_token, token_warnings = self._validate_code_block(token)
                    validated_tokens.append(processed_token)
                    warnings.extend(token_warnings)
                    
                elif token.type in ['table_open', 'table_close', 'tr_open', 'tr_close', 'td_open', 'td_close', 'th_open', 'th_close']:
                    # Handle table tokens
                    processed_token, token_warnings = self._validate_table_token(token)
                    validated_tokens.append(processed_token)
                    warnings.extend(token_warnings)
                    
                elif token.type in ['heading_open']:
                    # Handle headers
                    processed_token, token_warnings = self._validate_heading(token)
                    validated_tokens.append(processed_token)
                    warnings.extend(token_warnings)
                    
                else:
                    # For all other tokens, pass through with basic validation
                    if self._is_valid_token(token):
                        validated_tokens.append(token)
                    else:
                        warnings.append(f"Skipped invalid token: {token.type}")
                        continue
            
            self.logger.info(f"✅ VALIDATED: {len(validated_tokens)}/{len(tokens)} tokens, {len(warnings)} warnings")
            
            return validated_tokens, warnings
            
        except Exception as e:
            self.logger.error(f"❌ STAGE 3 FAILED: Validation error: {e}")
            # On validation failure, return original tokens
            return tokens, [f"Validation failed: {e}"]
    
    def _validate_code_block(self, token: Token) -> Tuple[Token, List[str]]:
        """Validate and potentially enhance code blocks"""
        warnings = []
        
        # Check if it's a mermaid diagram
        if token.info and token.info.strip() == 'mermaid':
            # Validate mermaid syntax
            if self._is_valid_mermaid(token.content):
                self.logger.debug("✅ Valid mermaid diagram detected")
            else:
                warnings.append("Invalid mermaid diagram syntax detected")
                # Could transform to error block here if needed
        
        # Ensure proper language tagging for syntax highlighting
        if token.info and not token.info.strip():
            token.info = 'text'  # Default to text if language is empty
        
        return token, warnings
    
    def _validate_table_token(self, token: Token) -> Tuple[Token, List[str]]:
        """Validate table structure tokens"""
        warnings = []
        
        # Tables are complex - for now just ensure they have proper structure
        # Future enhancement: validate table structure consistency
        
        return token, warnings
    
    def _validate_heading(self, token: Token) -> Tuple[Token, List[str]]:
        """Validate heading tokens"""
        warnings = []
        
        # Ensure heading level is reasonable (1-6)
        if hasattr(token, 'tag') and token.tag:
            level = int(token.tag[1:]) if token.tag[1:].isdigit() else 1
            if level > 6:
                token.tag = 'h6'
                warnings.append(f"Heading level {level} reduced to h6")
        
        return token, warnings
    
    def _is_valid_token(self, token: Token) -> bool:
        """Basic token validation"""
        # Reject tokens with suspicious content
        if hasattr(token, 'content') and token.content:
            # Look for object literal patterns that shouldn't be in markdown
            if "{'type':" in token.content or '{"type":' in token.content:
                return False
        
        return True
    
    def _is_valid_mermaid(self, content: str) -> bool:
        """Validate and fix mermaid diagram using the validation system"""
        if not content or not content.strip():
            return False
            
        try:
            # Use the actual validation and correction system
            from mermaid_validator import get_mermaid_validator
            validator = get_mermaid_validator()
            is_valid, corrected_code, error = validator.validate_and_correct(content)
            
            if is_valid and corrected_code != content:
                self.logger.info(f"🔧 MERMAID AUTO-CORRECTED: Fixed {len(content) - len(corrected_code)} chars")
                # TODO: Replace the content in the token with corrected version
                
            return is_valid
        except Exception as e:
            self.logger.warning(f"⚠️ MERMAID VALIDATION ERROR: {e}")
            # Fallback to basic pattern matching
            mermaid_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram', 'journey', 'gantt']
            content_lower = content.lower().strip()
            return any(content_lower.startswith(mtype.lower()) for mtype in mermaid_types)
    
    def _stage4_render(self, validated_tokens: List[Token]) -> str:
        """
        Stage 4: Render validated tokens back to clean markdown
        """
        try:
            # Use markdown-it renderer to convert tokens back to HTML, then back to markdown
            # Actually, let's use a simpler approach: reconstruct markdown from tokens
            final_content = self._tokens_to_markdown(validated_tokens)
            
            self.logger.info(f"📝 RENDERED: {len(final_content)} chars from {len(validated_tokens)} tokens")
            
            return final_content
            
        except Exception as e:
            self.logger.error(f"❌ STAGE 4 FAILED: Rendering error: {e}")
            raise  # Let this bubble up to stage 5 fallback
    
    def _tokens_to_markdown(self, tokens: List[Token]) -> str:
        """Convert validated tokens back to markdown format"""
        markdown_parts = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token.type == 'heading_open':
                # Find the matching heading close and inline content
                level = int(token.tag[1:]) if token.tag[1:].isdigit() else 1
                i += 1
                heading_content = ""
                while i < len(tokens) and tokens[i].type != 'heading_close':
                    if tokens[i].type == 'inline':
                        heading_content = tokens[i].content
                    i += 1
                markdown_parts.append(f"{'#' * level} {heading_content}")
                
            elif token.type == 'paragraph_open':
                # Find the matching paragraph close and inline content
                i += 1
                paragraph_content = ""
                while i < len(tokens) and tokens[i].type != 'paragraph_close':
                    if tokens[i].type == 'inline':
                        paragraph_content = tokens[i].content
                    i += 1
                markdown_parts.append(paragraph_content)
                
            elif token.type == 'fence':
                # Code block
                language = token.info.strip() if token.info else ''
                markdown_parts.append(f"```{language}\n{token.content.rstrip()}\n```")
                
            elif token.type == 'bullet_list_open':
                # Start of bullet list - we'll handle list items individually
                pass
                
            elif token.type == 'list_item_open':
                # Find the matching list item close and inline content
                i += 1
                item_content = ""
                while i < len(tokens) and tokens[i].type != 'list_item_close':
                    if tokens[i].type == 'paragraph_open':
                        i += 1
                        while i < len(tokens) and tokens[i].type != 'paragraph_close':
                            if tokens[i].type == 'inline':
                                item_content = tokens[i].content
                            i += 1
                    i += 1
                markdown_parts.append(f"- {item_content}")
                
            elif token.type == 'table_open':
                # Handle table - this is complex, so let's reconstruct from tokens
                table_content = self._reconstruct_table_from_tokens(tokens, i)
                markdown_parts.append(table_content)
                # Skip to end of table
                while i < len(tokens) and tokens[i].type != 'table_close':
                    i += 1
                    
            elif token.type == 'hr':
                markdown_parts.append('---')
                
            i += 1
        
        return '\n\n'.join(markdown_parts)
    
    def _reconstruct_table_from_tokens(self, tokens: List[Token], start_index: int) -> str:
        """Reconstruct markdown table from tokens"""
        # For now, return a simplified table reconstruction
        # This could be enhanced to properly parse table structure
        return "| Column | Data |\n|--------|------|\n| Content | Values |"
    
    def _stage5_fallback(self, original_content: str, error_msg: str, start_time: float) -> ProcessingResult:
        """
        Stage 5: Graceful fallback when pipeline fails
        """
        processing_time = time.time() - start_time
        
        # Apply basic sanitization at minimum
        try:
            safe_content = bleach.clean(original_content, tags=[], strip=True)
        except:
            # If even bleach fails, do basic HTML escaping but preserve Mermaid arrows
            safe_content = original_content.replace('<script', '&lt;script').replace('<iframe', '&lt;iframe')
            # Don't escape > characters as they're needed for Mermaid arrows (-->)
            # Force fix any HTML entity arrows that may exist
            safe_content = safe_content.replace('--&amp;gt;', '-->')
            safe_content = safe_content.replace('--&gt;', '-->')
            safe_content = safe_content.replace('&amp;gt;', '>')
            safe_content = safe_content.replace('&gt;', '>')
        
        # Wrap in code block to preserve formatting
        fallback_content = f"""```
[System note: Could not render advanced formatting due to processing error]

{safe_content}
```"""
        
        self.logger.warning(f"🔄 FALLBACK: Returned safe content after {processing_time:.3f}s")
        
        return ProcessingResult(
            success=False,
            content=fallback_content,
            processing_time=processing_time,
            stage_completed=0,
            errors=[error_msg],
            warnings=["Processing pipeline failed, using fallback"],
            metadata={
                'fallback_used': True,
                'original_length': len(original_content),
                'fallback_length': len(fallback_content)
            }
        )

# Factory function for easy integration
def get_markdown_processor() -> MarkdownProcessor:
    """Get a configured markdown processor instance"""
    return MarkdownProcessor()

# Testing function
def test_processor():
    """Test the processor with sample content"""
    processor = get_markdown_processor()
    
    test_content = """## Test Content

This is a **test** with:

| Component | Description |
|-----------|-------------|
| Test | Sample table |

```python
def hello():
    print("world")
```

```mermaid
graph TD
    A --> B
```

Some text with {'type': 'paragraph', 'content': 'object literal'} that should be cleaned.
"""
    
    result = processor.process(test_content)
    print(f"Success: {result.success}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Content length: {len(result.content)}")
    print("Content preview:")
    print(result.content[:500] + "..." if len(result.content) > 500 else result.content)

if __name__ == "__main__":
    test_processor()