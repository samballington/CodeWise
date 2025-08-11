#!/usr/bin/env python3
"""
Response Sanitization System for CodeWise

Handles LLM responses that may contain mixed JSON/text content and ensures
proper extraction of structured data for UI consumption.

Based on response_fix.md guidance for robust JSON handling.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """
    Sanitizes LLM responses to handle mixed JSON/text content and ensures
    proper extraction of structured data for UI consumption.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ResponseSanitizer")
    
    def extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text that may contain conversational content.
        
        Handles cases like:
        - "Here's the JSON: {...}"
        - "```json {...} ```"
        - "Sure! {...}"
        - Mixed content with JSON embedded
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Extracted JSON string or None if no JSON found
        """
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Case 1: Text is already pure JSON (starts and ends with braces)
        if text.startswith('{') and text.endswith('}'):
            self.logger.debug("🔍 SANITIZER: Text appears to be pure JSON")
            return text
        
        # Case 2: JSON wrapped in markdown code block
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_match = re.search(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if code_match:
            self.logger.debug("🔍 SANITIZER: Found JSON in markdown code block")
            return code_match.group(1).strip()
        
        # Case 3: JSON embedded in conversational text (first occurrence)
        # This regex finds the first complete JSON object (handles nested braces)
        brace_count = 0
        start_pos = None
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos is not None:
                    # Found complete JSON object
                    json_str = text[start_pos:i+1]
                    self.logger.debug(f"🔍 SANITIZER: Extracted JSON from position {start_pos}-{i+1}")
                    return json_str
        
        # Case 4: No JSON found
        self.logger.warning("⚠️ SANITIZER: No JSON object found in text")
        return None
    
    def validate_json_structure(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate that extracted JSON has expected structure for our application.
        
        Expected structure:
        {
          "response": {
            "sections": [...],
            "metadata": {...}
          }
        }
        
        Args:
            json_data: Parsed JSON data to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check top-level structure
            if not isinstance(json_data, dict):
                self.logger.warning("⚠️ SANITIZER: JSON is not a dictionary")
                return False
            
            if 'response' not in json_data:
                self.logger.warning("⚠️ SANITIZER: Missing 'response' key")
                return False
            
            response = json_data['response']
            if not isinstance(response, dict):
                self.logger.warning("⚠️ SANITIZER: 'response' is not a dictionary")
                return False
            
            # Check for sections (required)
            if 'sections' not in response:
                self.logger.warning("⚠️ SANITIZER: Missing 'sections' key")
                return False
            
            sections = response['sections']
            if not isinstance(sections, list):
                self.logger.warning("⚠️ SANITIZER: 'sections' is not a list")
                return False
            
            # Validate each section has required fields
            for i, section in enumerate(sections):
                if not isinstance(section, dict):
                    self.logger.warning(f"⚠️ SANITIZER: Section {i} is not a dictionary")
                    return False
                
                if 'type' not in section:
                    self.logger.warning(f"⚠️ SANITIZER: Section {i} missing 'type' field")
                    return False
            
            self.logger.info("✅ SANITIZER: JSON structure validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 SANITIZER: Structure validation error: {e}")
            return False
    
    def generate_readable_summary(self, sections: List[Dict[str, Any]]) -> str:
        """
        Convert structured sections to human-readable text for output field.
        
        Handles different section types:
        - headings: Convert to markdown-style headers
        - paragraphs: Plain text
        - diagrams: Placeholder text with format info
        - code blocks: Simple code representation
        - lists: Bullet or numbered lists
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Human-readable text summary
        """
        if not sections:
            return "No content available"
        
        readable_parts = []
        
        try:
            for section in sections:
                section_type = section.get('type', 'unknown')
                content = section.get('content', '')
                
                if section_type == 'heading':
                    level = section.get('level', 1)
                    prefix = '#' * min(level, 6)  # Limit to H6
                    readable_parts.append(f"{prefix} {content}")
                    
                elif section_type == 'paragraph':
                    readable_parts.append(content)
                    
                elif section_type == 'diagram':
                    diagram_format = section.get('format', 'unknown')
                    readable_parts.append(f"[{diagram_format.upper()} Diagram]")
                    # Note: Don't include actual diagram code in readable summary
                    
                elif section_type == 'code_block':
                    language = section.get('language', '')
                    readable_parts.append(f"```{language}\n{content}\n```")
                    
                elif section_type == 'list':
                    style = section.get('style', 'bullet')
                    items = section.get('items', [])
                    
                    if style == 'bullet':
                        for item in items:
                            readable_parts.append(f"• {item}")
                    else:  # numbered
                        for i, item in enumerate(items, 1):
                            readable_parts.append(f"{i}. {item}")
                            
                elif section_type == 'callout':
                    callout_style = section.get('style', 'info')
                    readable_parts.append(f"[{callout_style.upper()}] {content}")
                    
                elif section_type == 'table':
                    # Simple table representation
                    readable_parts.append("[TABLE]")
                    readable_parts.append(content)
                    
                else:
                    # Unknown type - just include content
                    readable_parts.append(content)
            
            result = '\n\n'.join(filter(None, readable_parts))
            
            if not result.strip():
                result = "Content processed successfully"
            
            self.logger.info(f"📝 SANITIZER: Generated readable summary ({len(result)} chars from {len(sections)} sections)")
            return result
            
        except Exception as e:
            self.logger.error(f"💥 SANITIZER: Error generating readable summary: {e}")
            return "Error processing response content"
    
    def sanitize_llm_response(self, raw_response: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Main sanitization function that processes raw LLM response.
        
        Process:
        1. Extract JSON from potentially mixed content
        2. Validate JSON structure
        3. Generate readable text summary
        4. Return sanitized results
        
        Args:
            raw_response: Raw response string from LLM
            
        Returns:
            Tuple of (success, readable_text, structured_data)
            - success: True if sanitization succeeded
            - readable_text: Human-readable summary for UI display
            - structured_data: Parsed JSON data for rich rendering (None if failed)
        """
        self.logger.info(f"🧼 SANITIZER: Processing LLM response ({len(raw_response)} chars)")
        
        try:
            # Step 1: Extract JSON from text
            json_string = self.extract_json_from_text(raw_response)
            if not json_string:
                self.logger.warning("⚠️ SANITIZER: No JSON found in response")
                return False, "Response format not recognized", None
            
            # Step 2: Parse JSON
            try:
                json_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ SANITIZER: JSON parsing failed: {e}")
                return False, "Invalid response format", None
            
            # Step 3: Validate structure
            if not self.validate_json_structure(json_data):
                self.logger.error("❌ SANITIZER: JSON structure validation failed")
                return False, "Response structure invalid", None
            
            # Step 4: Generate readable summary
            sections = json_data['response']['sections']
            readable_text = self.generate_readable_summary(sections)
            
            self.logger.info("✅ SANITIZER: Successfully sanitized LLM response")
            return True, readable_text, json_data
            
        except Exception as e:
            self.logger.error(f"💥 SANITIZER: Critical error during sanitization: {e}")
            return False, "Response processing failed", None


# Global instance for easy import
_sanitizer_instance = None

def get_response_sanitizer() -> ResponseSanitizer:
    """Get singleton ResponseSanitizer instance"""
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = ResponseSanitizer()
    return _sanitizer_instance