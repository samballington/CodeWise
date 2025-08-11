#!/usr/bin/env python3
"""
Response Consolidation System for CodeWise
Consolidates all response data sources into a single, comprehensive final_result message.
Prevents message fragmentation while preserving all response quality and functionality.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ResponseSource(Enum):
    """Source of response data for priority handling"""
    STRUCTURED = "structured"           # Highest priority - JSON prompt format
    FORMATTED = "formatted"            # Medium priority - response_formatter output  
    RAW = "raw"                       # Lowest priority - plain text fallback
    SYNTHESIS = "synthesis"           # Enhanced - from synthesis stage
    MAX_ITERATIONS = "max_iterations" # Special case - iteration limit reached


@dataclass
class ResponseData:
    """Container for all possible response data"""
    # Core content
    raw_output: str
    source: ResponseSource
    
    # Enhanced content (optional)
    structured_response: Optional[Dict[str, Any]] = None
    formatted_response: Optional[Dict[str, Any]] = None
    
    # Metadata
    execution_metadata: Optional[Dict[str, Any]] = None
    synthesis_triggered: bool = False
    max_iterations_reached: bool = False
    
    # Error information
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ResponseConsolidator:
    """
    Consolidates multiple response sources into a single comprehensive response.
    
    Design Principles:
    1. Preserve ALL data - never lose information
    2. Priority-based selection - structured > formatted > raw
    3. Metadata aggregation - combine all available metadata
    4. Error tracking - log all errors without failing
    5. Single output - one final_result message per query
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ResponseConsolidator")
        self.response_data: List[ResponseData] = []
        self.consolidated = False
    
    def add_response_data(
        self,
        raw_output: str,
        source: ResponseSource,
        structured_response: Optional[Dict[str, Any]] = None,
        formatted_response: Optional[Dict[str, Any]] = None,
        execution_metadata: Optional[Dict[str, Any]] = None,
        synthesis_triggered: bool = False,
        max_iterations_reached: bool = False,
        error: Optional[str] = None
    ):
        """Add response data from any source"""
        response = ResponseData(
            raw_output=raw_output,
            source=source,
            structured_response=structured_response,
            formatted_response=formatted_response,
            execution_metadata=execution_metadata,
            synthesis_triggered=synthesis_triggered,
            max_iterations_reached=max_iterations_reached
        )
        
        if error:
            response.errors.append(error)
        
        self.response_data.append(response)
        self.logger.info(f"📦 CONSOLIDATOR: Added {source.value} response data ({len(raw_output)} chars)")
    
    def add_error(self, error: str, source: ResponseSource = ResponseSource.RAW):
        """Add error information"""
        if self.response_data:
            self.response_data[-1].errors.append(error)
        else:
            # No response data yet, create error-only entry
            self.add_response_data("", source, error=error)
        
        self.logger.warning(f"⚠️ CONSOLIDATOR: Added error from {source.value}: {error}")
    
    async def validate_mermaid_diagrams(
        self, 
        response_data: Dict[str, Any], 
        original_query: str = "",
        llm_provider=None
    ) -> Dict[str, Any]:
        """
        Validate any Mermaid diagrams in structured response and attempt regeneration if needed.
        
        Args:
            response_data: Response data containing potential Mermaid diagrams
            original_query: Original user query for regeneration context
            llm_provider: LLM provider for regeneration
            
        Returns:
            Response data with validation results and corrected diagrams
        """
        try:
            from mermaid_validator import get_mermaid_validator
            validator = get_mermaid_validator()
            
            # Check if response has structured sections with diagrams
            if 'structured_response' in response_data:
                structured = response_data['structured_response']
                if isinstance(structured, dict) and 'response' in structured:
                    sections = structured.get('response', {}).get('sections', [])
                    
                    for section in sections:
                        if isinstance(section, dict) and section.get('type') == 'diagram' and section.get('format') == 'mermaid':
                            content = section.get('content', '')
                            if content:
                                self.logger.info("🔍 MERMAID: Validating diagram with regeneration support")
                                
                                # Use enhanced validation with regeneration
                                is_valid, final_code, error, user_message = await validator.validate_and_regenerate(
                                    mermaid_code=content,
                                    original_query=original_query,
                                    llm_provider=llm_provider
                                )
                                
                                # Update section with results
                                section['content'] = final_code
                                if user_message:
                                    # Add user message as a note in the section
                                    section['regeneration_note'] = user_message
                                
                                if is_valid:
                                    self.logger.info("✅ MERMAID: Diagram validated successfully")
                                else:
                                    self.logger.warning(f"⚠️ MERMAID: Diagram validation failed: {error}")
            
            return response_data
            
        except ImportError:
            self.logger.warning("⚠️ MERMAID VALIDATION: Validator not available, skipping validation")
            return response_data
        except Exception as e:
            self.logger.error(f"💥 MERMAID VALIDATION: Validation failed - {e}")
            return response_data
    
    def _normalize_response_format(self, raw_output: str, source: ResponseSource) -> List[Dict[str, Any]]:
        """
        Normalize any LLM response format to unified sections array.
        
        This is the key fix: convert all input formats to consistent internal format
        before consolidation to eliminate format ambiguity.
        
        Args:
            raw_output: Raw response from LLM
            source: Source type (STRUCTURED, FORMATTED, RAW)
            
        Returns:
            List of normalized sections
        """
        try:
            if source == ResponseSource.STRUCTURED:
                # Handle: {"response": {"sections": [...], "metadata": {...}}}
                try:
                    import json
                    parsed = json.loads(raw_output)
                    if isinstance(parsed, dict) and 'response' in parsed:
                        sections = parsed['response'].get('sections', [])
                        if isinstance(sections, list):
                            self.logger.info(f"🔧 FORMAT NORMALIZER: Extracted {len(sections)} sections from STRUCTURED response")
                            return sections
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    self.logger.warning(f"⚠️ FORMAT NORMALIZER: STRUCTURED parsing failed: {e}")
            
            elif source == ResponseSource.FORMATTED:
                # Handle: {"answer": "text...", "sections": [...], "confidence": 0.9}
                try:
                    import json
                    parsed = json.loads(raw_output)
                    if isinstance(parsed, dict) and 'sections' in parsed:
                        sections = parsed.get('sections', [])
                        if isinstance(sections, list):
                            self.logger.info(f"🔧 FORMAT NORMALIZER: Extracted {len(sections)} sections from FORMATTED response")
                            return sections
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    self.logger.warning(f"⚠️ FORMAT NORMALIZER: FORMATTED parsing failed: {e}")
            
            elif source in [ResponseSource.RAW, ResponseSource.SYNTHESIS, ResponseSource.MAX_ITERATIONS]:
                # Handle: plain text or any fallback case
                self.logger.info("🔧 FORMAT NORMALIZER: Converting RAW text to text section")
                return [{
                    "type": "text",
                    "content": raw_output.strip()
                }]
            
            # Fallback for any unhandled case
            self.logger.warning(f"⚠️ FORMAT NORMALIZER: Fallback - converting {source.value} to text section")
            return [{
                "type": "text", 
                "content": raw_output.strip() if raw_output else "No content available"
            }]
            
        except Exception as e:
            self.logger.error(f"💥 FORMAT NORMALIZER: Critical error normalizing {source.value}: {e}")
            return [{
                "type": "text",
                "content": f"Error processing response: {str(e)}"
            }]
    
    def _generate_readable_output(self, sections: List[Dict[str, Any]]) -> str:
        """
        Generate readable text output from normalized sections for frontend display.
        This maintains the frontend contract while providing internal structure.
        """
        readable_parts = []
        
        try:
            for section in sections:
                section_type = section.get('type', 'text')
                content = section.get('content', '')
                
                if section_type == 'heading':
                    level = section.get('level', 1)
                    prefix = '#' * level
                    readable_parts.append(f"{prefix} {content}")
                    
                elif section_type == 'paragraph':
                    readable_parts.append(content)
                    
                elif section_type == 'diagram':
                    diagram_format = section.get('format', 'unknown')
                    diagram_content = section.get('content', '')
                    readable_parts.append(f"```{diagram_format.lower()}\n{diagram_content}\n```")
                    # Include the diagram code as markdown code block for rendering
                    
                elif section_type == 'code':
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
                            
                else:  # text or unknown
                    readable_parts.append(content)
            
            result = '\n\n'.join(filter(None, readable_parts))
            self.logger.info(f"📝 READABLE OUTPUT: Generated {len(result)} chars from {len(sections)} sections")
            return result
            
        except Exception as e:
            self.logger.error(f"💥 READABLE OUTPUT: Generation failed: {e}")
            # Fallback to joining all content
            return '\n\n'.join(section.get('content', '') for section in sections)
    
    async def consolidate(self, original_query: str = "", llm_provider=None) -> Dict[str, Any]:
        """
        Consolidate all response data into a single comprehensive response.
        
        Priority Order:
        1. STRUCTURED (JSON prompt) - most complete
        2. FORMATTED (response_formatter) - enhanced with metadata  
        3. RAW (plain text) - basic fallback
        
        Always preserves:
        - Highest quality content
        - All metadata
        - Error information
        - Source tracking
        """
        if self.consolidated:
            self.logger.warning("🔄 CONSOLIDATOR: Already consolidated, returning cached result")
            return self._get_cached_result()
        
        self.logger.info(f"🔧 CONSOLIDATOR: Starting consolidation of {len(self.response_data)} data sources")
        
        if not self.response_data:
            self.logger.error("💥 CONSOLIDATOR: No response data to consolidate")
            return self._create_error_response("No response data available")
        
        # Sort by priority: STRUCTURED > FORMATTED > RAW
        priority_order = [
            ResponseSource.STRUCTURED,
            ResponseSource.SYNTHESIS,  
            ResponseSource.FORMATTED,
            ResponseSource.RAW,
            ResponseSource.MAX_ITERATIONS
        ]
        
        sorted_data = sorted(
            self.response_data,
            key=lambda x: priority_order.index(x.source) if x.source in priority_order else len(priority_order)
        )
        
        # Use highest priority data as primary
        primary_data = sorted_data[0]
        self.logger.info(f"🎯 CONSOLIDATOR: Using {primary_data.source.value} as primary response")
        
        # Consolidate metadata from all sources
        consolidated_metadata = self._merge_metadata([d.execution_metadata for d in self.response_data])
        
        # Collect all errors
        all_errors = []
        for data in self.response_data:
            all_errors.extend(data.errors)
        
        # NEW LOGIC: Handle structured responses differently
        if primary_data.source == ResponseSource.STRUCTURED:
            # Use structured data directly - no sanitization needed
            structured_data = primary_data.structured_response
            
            if structured_data:
                self.logger.info("📊 CONSOLIDATOR: Using structured response data directly")
                
                # Generate readable output from structured data
                readable_output = self._generate_readable_output(structured_data.get('response', {}).get('sections', []))
                
                final_response = {
                    "type": "final_result",
                    "output": readable_output,
                    "structured_response": structured_data,
                    "consolidation_metadata": {
                        "primary_source": primary_data.source.value,
                        "total_sources": len(self.response_data),
                        "sources_used": [d.source.value for d in self.response_data],
                        "synthesis_triggered": any(d.synthesis_triggered for d in self.response_data),
                        "max_iterations_reached": any(d.max_iterations_reached for d in self.response_data),
                        "errors_encountered": len(all_errors),
                        "execution_metadata": consolidated_metadata,
                        "response_type": "structured"
                    }
                }
            else:
                # Fallback for missing structured data
                self.logger.error("💥 CONSOLIDATOR: Structured response missing data")
                final_response = {
                    "type": "final_result",
                    "output": "Error: Structured response data is missing",
                    "consolidation_metadata": {
                        "primary_source": primary_data.source.value,
                        "error": "Missing structured data"
                    }
                }
        else:
            # Handle non-structured responses (existing logic)
            self.logger.info(f"📊 CONSOLIDATOR: Using {primary_data.source.value} response with sanitization")
            
            # Apply sanitization for non-structured responses
            from response_sanitizer import get_response_sanitizer
            sanitizer = get_response_sanitizer()
            
            success, readable_text, structured_data = sanitizer.sanitize_llm_response(
                primary_data.raw_output
            )
            
            if success and structured_data:
                # Sanitizer found structured data in markdown response
                output_content = readable_text
                enhanced_content = {"structured_response": structured_data}
            else:
                # Pure markdown/text response
                output_content = primary_data.raw_output
                enhanced_content = self._select_best_enhanced_content(sorted_data)
            
            final_response = {
                "type": "final_result",
                "output": output_content,
                **enhanced_content,
                "consolidation_metadata": {
                    "primary_source": primary_data.source.value,
                    "total_sources": len(self.response_data),
                    "sources_used": [d.source.value for d in self.response_data],
                    "synthesis_triggered": any(d.synthesis_triggered for d in self.response_data),
                    "max_iterations_reached": any(d.max_iterations_reached for d in self.response_data),
                    "errors_encountered": len(all_errors),
                    "execution_metadata": consolidated_metadata,
                    "sanitization_applied": success,
                    "response_type": "markdown"
                }
            }
        
        # Apply Mermaid validation and enhancement
        final_response = await self.validate_mermaid_diagrams(
            final_response, 
            original_query=original_query,
            llm_provider=llm_provider
        )
        
        # Add error information if present (for debugging)
        if all_errors:
            final_response["consolidation_metadata"]["errors"] = all_errors
            self.logger.warning(f"⚠️ CONSOLIDATOR: Consolidated response contains {len(all_errors)} errors")
        
        self.consolidated = True
        self._cached_result = final_response
        
        self.logger.info(f"✅ CONSOLIDATOR: Consolidation complete - {len(final_response.get('output', ''))} chars, "
                        f"primary_source={primary_data.source.value}")
        
        return final_response
    
    def _select_best_enhanced_content(self, sorted_data: List[ResponseData]) -> Dict[str, Any]:
        """Select the best enhanced content from available sources - SIMPLE VERSION"""
        enhanced_content = {}
        
        # Try to get structured response first
        for data in sorted_data:
            if data.structured_response:
                enhanced_content["structured_response"] = data.structured_response
                self.logger.info("📋 CONSOLIDATOR: Using structured_response")
                break
        
        # If no structured response, try formatted response
        if "structured_response" not in enhanced_content:
            for data in sorted_data:
                if data.formatted_response:
                    enhanced_content["formatted_response"] = data.formatted_response
                    self.logger.info("📊 CONSOLIDATOR: Using formatted_response")
                    break
        
        return enhanced_content
    
    def _merge_metadata(self, metadata_list: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """Merge metadata from all sources"""
        merged = {}
        
        for metadata in metadata_list:
            if metadata:
                for key, value in metadata.items():
                    if key not in merged:
                        merged[key] = value
                    elif key == 'total_time' and isinstance(value, (int, float)):
                        # For time, use the maximum (most complete execution)
                        merged[key] = max(merged.get(key, 0), value)
                    elif key in ['iterations', 'tool_calls'] and isinstance(value, (int, float)):
                        # For counts, use the maximum
                        merged[key] = max(merged.get(key, 0), value)
        
        return merged
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response when consolidation fails"""
        return {
            "type": "final_result",
            "output": f"Error: {error_message}",
            "consolidation_metadata": {
                "primary_source": "error",
                "total_sources": 0,
                "sources_used": [],
                "error": error_message
            }
        }
    
    def _get_cached_result(self) -> Dict[str, Any]:
        """Return cached consolidated result"""
        return getattr(self, '_cached_result', self._create_error_response("Consolidation cache missing"))
    
    def has_data(self) -> bool:
        """Check if any response data has been added"""
        return len(self.response_data) > 0
    
    def get_primary_source(self) -> Optional[ResponseSource]:
        """Get the primary source that will be used in consolidation"""
        if not self.response_data:
            return None
        
        priority_order = [
            ResponseSource.STRUCTURED,
            ResponseSource.SYNTHESIS,
            ResponseSource.FORMATTED,
            ResponseSource.RAW,
            ResponseSource.MAX_ITERATIONS
        ]
        
        for source in priority_order:
            for data in self.response_data:
                if data.source == source:
                    return source
        
        return self.response_data[0].source if self.response_data else None