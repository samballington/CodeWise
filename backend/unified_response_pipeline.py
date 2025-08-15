#!/usr/bin/env python3
"""
Unified Response Pipeline - Modular response processing that always produces
consistent output regardless of AI response format or prompting changes.

UPDATED: Now uses handbook-based markdown processing pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import re
import logging

# NEW: Import handbook-based markdown processor
from markdown_processor import get_markdown_processor

logger = logging.getLogger(__name__)

class ContentType(Enum):
    STRUCTURED_JSON = "structured_json"
    MARKDOWN = "markdown" 
    MIXED = "mixed"
    INVALID = "invalid"

class PipelineType(Enum):
    STRUCTURED = "structured"
    FORMATTED = "formatted"
    FALLBACK = "fallback"

@dataclass
class AnalysisResult:
    content_type: ContentType
    quality_score: float
    recommended_pipeline: PipelineType
    issues: List[str]
    confidence: float

@dataclass
class ProcessingInfo:
    pipeline_used: PipelineType
    processing_time: float
    errors: List[str]
    fallback_triggered: bool

@dataclass
class UnifiedResponse:
    content: str
    metadata: Dict[str, Any]
    structured_data: Optional[Dict] = None
    processing_info: Optional[ProcessingInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        from json_encoder import to_json_dict
        return to_json_dict(self)

class ResponseAnalyzer:
    """Analyzes AI responses to determine optimal processing strategy"""
    
    def analyze(self, raw_response: str) -> AnalysisResult:
        """Analyze response and recommend processing strategy"""
        
        issues = []
        quality_score = 0.0
        
        # Check for JSON structure
        has_json = self._detect_json_structure(raw_response)
        
        # Check for object literals (the problem we're fixing)
        has_object_literals = self._detect_object_literals(raw_response)
        if has_object_literals:
            issues.append("Object literals detected in text")
        
        # Check content quality
        content_quality = self._assess_content_quality(raw_response)
        quality_score += content_quality
        
        # Determine content type
        if has_json and not has_object_literals:
            content_type = ContentType.STRUCTURED_JSON
            recommended_pipeline = PipelineType.STRUCTURED
        elif has_object_literals or quality_score < 0.3:
            content_type = ContentType.INVALID
            recommended_pipeline = PipelineType.FALLBACK
        else:
            content_type = ContentType.MARKDOWN
            recommended_pipeline = PipelineType.FORMATTED
        
        return AnalysisResult(
            content_type=content_type,
            quality_score=quality_score,
            recommended_pipeline=recommended_pipeline,
            issues=issues,
            confidence=min(quality_score, 1.0)
        )
    
    def _detect_json_structure(self, text: str) -> bool:
        """Detect if response contains valid JSON structure"""
        try:
            # Look for JSON-like patterns
            json_patterns = [
                r'\{"response":\s*\{',
                r'\{"metadata":\s*\{',
                r'\{"sections":\s*\['
            ]
            return any(re.search(pattern, text) for pattern in json_patterns)
        except:
            return False
    
    def _detect_object_literals(self, text: str) -> bool:
        """Detect problematic object literals in text"""
        # Patterns that break the frontend (with and without bullet points)
        patterns = [
            r"•\s*\{'type':\s*'paragraph',\s*'content':",  # With bullet
            r"\{'type':\s*'paragraph',\s*'content':",      # Without bullet
            r"•\s*\{'type':\s*'[^']+',\s*'content':",      # Any type with bullet
            r"\{'type':\s*'[^']+',\s*'content':"           # Any type without bullet
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _assess_content_quality(self, text: str) -> float:
        """Assess overall content quality"""
        score = 0.0
        
        # Length check
        if len(text) > 500:
            score += 0.3
        
        # Structure check
        if '##' in text or '###' in text:
            score += 0.2
            
        # Code blocks
        if '```' in text:
            score += 0.1
            
        # Completeness check
        if len(text) > 1000 and 'architecture' in text.lower():
            score += 0.4
            
        return min(score, 1.0)

class UnifiedConverter:
    """Converts any response format to standard UnifiedResponse"""
    
    def convert_structured_response(self, structured_data: Dict, raw_text: str) -> UnifiedResponse:
        """Convert structured JSON response to unified format"""
        
        try:
            # Extract sections from structured data
            response_data = structured_data.get('response', {})
            sections = response_data.get('sections', [])
            
            # Convert sections to clean markdown
            markdown_content = self._sections_to_markdown(sections)
            
            # Extract metadata
            metadata = response_data.get('metadata', {})
            
            return UnifiedResponse(
                content=markdown_content,
                metadata=metadata,
                structured_data=structured_data,
                processing_info=ProcessingInfo(
                    pipeline_used=PipelineType.STRUCTURED,
                    processing_time=0.0,
                    errors=[],
                    fallback_triggered=False
                )
            )
            
        except Exception as e:
            logger.error(f"Structured conversion failed: {e}")
            return self._fallback_conversion(raw_text, f"Structured parsing error: {e}")
    
    def convert_formatted_response(self, formatted_data: Dict, raw_text: str) -> UnifiedResponse:
        """Convert response_formatter output to unified format"""
        
        try:
            # Extract the main answer content
            content = formatted_data.get('answer', raw_text)
            
            # Clean up template artifacts
            content = self._clean_template_artifacts(content)
            
            # Extract metadata
            metadata = {
                'query_type': formatted_data.get('query_type', 'general'),
                'confidence': formatted_data.get('confidence_score', 0.5)
            }
            
            return UnifiedResponse(
                content=content,
                metadata=metadata,
                structured_data=None,
                processing_info=ProcessingInfo(
                    pipeline_used=PipelineType.FORMATTED,
                    processing_time=0.0,
                    errors=[],
                    fallback_triggered=False
                )
            )
            
        except Exception as e:
            logger.error(f"Formatted conversion failed: {e}")
            return self._fallback_conversion(raw_text, f"Formatted parsing error: {e}")
    
    def _sections_to_markdown(self, sections: List[Dict]) -> str:
        """Convert structured sections to clean markdown"""
        
        markdown_parts = []
        
        for section in sections:
            section_type = section.get('type', '')
            content = section.get('content', '')
            
            if section_type == 'heading':
                level = section.get('level', 1)
                markdown_parts.append(f"{'#' * level} {content}")
                
            elif section_type == 'paragraph':
                # FIX: Extract content from object if needed
                if isinstance(content, dict) and 'content' in content:
                    content = content['content']
                markdown_parts.append(content)
                
            elif section_type == 'list':
                items = section.get('items', [])
                for item in items:
                    markdown_parts.append(f"- {item}")
                    
            elif section_type == 'code':
                language = section.get('language', '')
                markdown_parts.append(f"```{language}\n{content}\n```")
                
            elif section_type == 'diagram':
                diagram_format = section.get('format', 'mermaid')
                markdown_parts.append(f"```{diagram_format}\n{content}\n```")
                
            else:
                # Fallback: treat as paragraph
                markdown_parts.append(str(content))
        
        return '\n\n'.join(markdown_parts)
    
    def _clean_template_artifacts(self, content: str) -> str:
        """Clean up template formatting artifacts"""
        
        # Remove template variable remnants
        content = re.sub(r'\{[^}]+\}', '', content)
        
        # Fix nested markdown artifacts
        content = re.sub(r'```markdown\n```markdown\n', '```markdown\n', content)
        
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _fallback_conversion(self, raw_text: str, error: str) -> UnifiedResponse:
        """Fallback conversion when other methods fail"""
        
        # Use handbook-based processing for fallback too
        try:
            markdown_processor = get_markdown_processor()
            processing_result = markdown_processor.process(raw_text)
            cleaned_text = processing_result.content
            errors = [error] + processing_result.errors
        except Exception as fallback_error:
            # If even handbook processing fails, use basic text safety only
            logger.error(f"❌ FALLBACK PROCESSING FAILED: {fallback_error}")
            # Apply minimal safety measures without destructive cleaning
            cleaned_text = raw_text.replace('<script', '&lt;script').replace('<iframe', '&lt;iframe')
            errors = [error, f"Handbook fallback failed: {fallback_error}"]
        
        return UnifiedResponse(
            content=cleaned_text,
            metadata={'query_type': 'general', 'confidence': 0.3},
            structured_data=None,
            processing_info=ProcessingInfo(
                pipeline_used=PipelineType.FALLBACK,
                processing_time=0.0,
                errors=errors,
                fallback_triggered=True
            )
        )
    
    # DEPRECATED: Old nuclear cleaning method - replaced by handbook-based processing
    def _clean_object_literals(self, text: str) -> str:
        """DEPRECATED: This method has been replaced by the handbook-based markdown processor"""
        logger.warning("⚠️ DEPRECATED: _clean_object_literals called - should use handbook processor instead")
        # Return text unchanged - all cleaning now handled by handbook processor
        return text

class ResponseValidator:
    """Validates unified responses for quality and correctness"""
    
    def validate(self, response: UnifiedResponse) -> Tuple[bool, List[str]]:
        """Validate response and return (is_valid, issues)"""
        
        issues = []
        
        # Comprehensive object literal detection
        object_literal_patterns = [
            r"\{'type':\s*'[^']+',\s*'content':",
            r"\{'type':\s*\"[^\"]+\",\s*'content':",
            r"•\s*\{'type':",
            r"{\s*\"type\":\s*\"[^\"]+\"",
            r"{\s*'type':\s*'[^']+'"
        ]
        
        # Additional patterns for malformed JSON
        malformed_json_patterns = [
            r'##\s+\w+.*,\s*"sections":\s*\[',  # Mixed markdown/JSON
            r',\s*"[^"]*":\s*\[',               # Orphaned JSON arrays
            r'^\s*,\s*$',                       # Standalone commas
            r',\s*\n\s*,',                      # Multiple commas
        ]
        
        for pattern in object_literal_patterns:
            if re.search(pattern, response.content):
                issues.append(f"Object literals detected: {pattern}")
                break
                
        for pattern in malformed_json_patterns:
            if re.search(pattern, response.content, re.MULTILINE | re.DOTALL):
                issues.append(f"Malformed JSON detected: {pattern}")
                break
        
        # Check minimum content length
        if len(response.content) < 50:
            issues.append("Content too short")
        
        # Check for broken markdown
        if response.content.count('```') % 2 != 0:
            issues.append("Unmatched code block delimiters")
        
        # Check for template artifacts
        if '{' in response.content and '}' in response.content:
            template_patterns = [
                r'\{[^}]*variable[^}]*\}',
                r'\{[^}]*\$[^}]*\}',
                r'\{[^}]*undefined[^}]*\}'
            ]
            for pattern in template_patterns:
                if re.search(pattern, response.content):
                    issues.append("Template artifacts detected")
                    break
        
        # Check for Python object representations
        if re.search(r"<[^>]*object at 0x[^>]*>", response.content):
            issues.append("Python object representations detected")
            
        # Check for JSON serialization errors
        if "not JSON serializable" in response.content:
            issues.append("JSON serialization error traces in content")
        
        return len(issues) == 0, issues

class UnifiedResponsePipeline:
    """Main pipeline coordinator"""
    
    def __init__(self):
        self.analyzer = ResponseAnalyzer()
        self.converter = UnifiedConverter()
        self.validator = ResponseValidator()
    
    def process_response(
        self, 
        raw_response: str, 
        tool_results: List[Dict],
        query: str,
        execution_metadata: Dict
    ) -> UnifiedResponse:
        """
        Pure markdown pipeline: clean, validate, and pass through markdown content
        """
        
        logger.info(f"🔄 PURE MARKDOWN PIPELINE: Processing response ({len(raw_response)} chars)")
        
        # STEP 1: Clean and validate markdown
        logger.info("🛡️ MARKDOWN PROCESSING: Ensuring pure markdown output")
        markdown_processor = get_markdown_processor()
        processing_result = markdown_processor.process(raw_response)
        
        if processing_result.success:
            clean_content = processing_result.content
            logger.info(f"✅ MARKDOWN SUCCESS: {len(raw_response)} -> {len(clean_content)} chars in {processing_result.processing_time:.3f}s")
        else:
            logger.warning(f"⚠️ MARKDOWN PARTIAL: Using fallback content")
            clean_content = processing_result.content
        
        # STEP 2: Create unified response with markdown content
        unified_response = UnifiedResponse(
            content=clean_content,
            metadata={
                'query_type': 'general',
                'confidence': 0.9,  # High confidence for clean markdown
                'format': 'markdown',
                'processing_time': processing_result.processing_time
            },
            structured_data=None,  # Pure markdown approach
            processing_info=ProcessingInfo(
                pipeline_used=PipelineType.FORMATTED,
                processing_time=processing_result.processing_time,
                errors=processing_result.errors,
                fallback_triggered=not processing_result.success
            )
        )
        
        # STEP 3: Final validation for monitoring
        is_valid, issues = self.validator.validate(unified_response)
        if not is_valid:
            logger.warning(f"⚠️ MARKDOWN VALIDATION: {issues}")
            unified_response.processing_info.errors.extend(issues)
        else:
            logger.info("✅ MARKDOWN VALIDATION: Pure markdown validated successfully")
        
        if processing_result.warnings:
            logger.warning(f"⚠️ MARKDOWN WARNINGS: {processing_result.warnings}")
        
        logger.info(f"✅ PURE MARKDOWN PIPELINE: Success, {len(clean_content)} chars")
        return unified_response
    
    def _convert_with_fallbacks(
        self, 
        raw_response: str, 
        analysis: AnalysisResult,
        tool_results: List[Dict],
        query: str,
        execution_metadata: Dict
    ) -> UnifiedResponse:
        """Try conversion strategies with fallbacks"""
        
        # Strategy 1: Try structured conversion
        if analysis.content_type == ContentType.STRUCTURED_JSON:
            try:
                from json_prompt_schema import parse_json_prompt
                structured_data = parse_json_prompt(raw_response)
                if structured_data:
                    return self.converter.convert_structured_response(structured_data.model_dump(), raw_response)
            except Exception as e:
                logger.warning(f"Structured conversion failed: {e}")
        
        # Strategy 2: Try formatted conversion  
        if analysis.content_type in [ContentType.MARKDOWN, ContentType.MIXED]:
            try:
                from response_formatter import ResponseFormatter
                formatter = ResponseFormatter()
                formatted_response = formatter.format_response(raw_response, tool_results, query, execution_metadata)
                return self.converter.convert_formatted_response(formatted_response.to_dict(), raw_response)
            except Exception as e:
                logger.warning(f"Formatted conversion failed: {e}")
        
        # Strategy 3: Fallback conversion
        logger.info("Using fallback conversion")
        return self.converter._fallback_conversion(raw_response, "Primary conversion strategies failed")

# Global instance
_unified_pipeline = None

def get_unified_pipeline() -> UnifiedResponsePipeline:
    """Get global unified pipeline instance"""
    global _unified_pipeline
    if _unified_pipeline is None:
        _unified_pipeline = UnifiedResponsePipeline()
    return _unified_pipeline