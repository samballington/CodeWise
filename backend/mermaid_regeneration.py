#!/usr/bin/env python3
"""
Mermaid LLM Regeneration System
Provides intelligent regeneration of invalid Mermaid diagrams using LLM providers.
"""

import logging
import time
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of validation error types"""
    SYNTAX = "syntax"
    STRUCTURE = "structure"
    SEMANTIC = "semantic"
    FORMAT = "format"
    UNKNOWN = "unknown"


class RegenerationStatus(Enum):
    """Status of regeneration attempts"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    DISABLED = "disabled"


@dataclass
class RegenerationConfig:
    """Configuration for Mermaid diagram regeneration"""
    enabled: bool = True
    max_attempts: int = 2
    timeout_seconds: int = 30
    include_syntax_rules: bool = True
    include_examples: bool = True
    fallback_to_text: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'RegenerationConfig':
        """Create configuration from environment variables"""
        import os
        return cls(
            enabled=os.getenv('MERMAID_REGENERATION_ENABLED', 'true').lower() == 'true',
            max_attempts=int(os.getenv('MERMAID_REGENERATION_MAX_ATTEMPTS', '2')),
            timeout_seconds=int(os.getenv('MERMAID_REGENERATION_TIMEOUT', '30')),
            include_syntax_rules=os.getenv('MERMAID_INCLUDE_SYNTAX_RULES', 'true').lower() == 'true',
            include_examples=os.getenv('MERMAID_INCLUDE_EXAMPLES', 'true').lower() == 'true',
            fallback_to_text=os.getenv('MERMAID_FALLBACK_TO_TEXT', 'true').lower() == 'true',
            log_level=os.getenv('MERMAID_REGENERATION_LOG_LEVEL', 'INFO')
        )


@dataclass
class ValidationErrorContext:
    """Structured information about validation errors"""
    error_message: str
    error_type: ErrorType
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    severity: str = "critical"
    
    @classmethod
    def from_validation_error(cls, error_message: str) -> 'ValidationErrorContext':
        """Create error context from validation error message"""
        error_type = cls._classify_error(error_message)
        line_number = cls._extract_line_number(error_message)
        suggested_fix = cls._generate_suggested_fix(error_message, error_type)
        
        return cls(
            error_message=error_message,
            error_type=error_type,
            line_number=line_number,
            suggested_fix=suggested_fix,
            severity="critical"
        )
    
    @staticmethod
    def _classify_error(error_message: str) -> ErrorType:
        """Classify error type based on error message"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in [
            'uppercase', 'syntax', 'parse error', 'invalid character', 'unexpected token'
        ]):
            return ErrorType.SYNTAX
        elif any(keyword in error_lower for keyword in [
            'unbalanced', 'missing end', 'graph declaration', 'subgraph'
        ]):
            return ErrorType.STRUCTURE
        elif any(keyword in error_lower for keyword in [
            'node reference', 'circular', 'dependency', 'relationship'
        ]):
            return ErrorType.SEMANTIC
        elif any(keyword in error_lower for keyword in [
            'format', 'encoding', 'ampersand', 'quotes'
        ]):
            return ErrorType.FORMAT
        else:
            return ErrorType.UNKNOWN
    
    @staticmethod
    def _extract_line_number(error_message: str) -> Optional[int]:
        """Extract line number from error message if present"""
        line_match = re.search(r'line\s+(\d+)', error_message, re.IGNORECASE)
        return int(line_match.group(1)) if line_match else None
    
    @staticmethod
    def _generate_suggested_fix(error_message: str, error_type: ErrorType) -> Optional[str]:
        """Generate suggested fix based on error message and type"""
        error_lower = error_message.lower()
        
        if 'uppercase' in error_lower:
            return "Convert direction to uppercase (e.g., 'graph TD' instead of 'graph td')"
        elif 'missing end' in error_lower:
            return "Add 'end' statement to close subgraph"
        elif 'graph declaration' in error_lower:
            return "Add graph declaration at the beginning (e.g., 'graph TD')"
        elif 'ampersand' in error_lower:
            return "Escape ampersands in labels using '&amp;'"
        elif 'quotes' in error_lower:
            return "Quote labels containing special characters"
        else:
            return None


@dataclass
class RegenerationResult:
    """Result of diagram regeneration attempt"""
    status: RegenerationStatus
    final_diagram: Optional[str] = None
    attempts_made: int = 0
    errors_encountered: List[str] = field(default_factory=list)
    regeneration_time: float = 0.0
    fallback_used: bool = False
    fallback_content: Optional[str] = None
    user_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if regeneration was successful"""
        return self.status == RegenerationStatus.SUCCESS
    
    def add_error(self, error: str):
        """Add error to the list of encountered errors"""
        self.errors_encountered.append(error)
        logger.warning(f"Regeneration error: {error}")
    
    def set_fallback(self, content: str, message: str):
        """Set fallback content and message"""
        self.fallback_used = True
        self.fallback_content = content
        self.user_message = message


@dataclass
class RegenerationRequest:
    """Request for diagram regeneration"""
    original_query: str
    invalid_diagram: str
    validation_error: ValidationErrorContext
    attempt_number: int = 1
    previous_attempts: List[str] = field(default_factory=list)
    user_context: Optional[Dict[str, Any]] = None
    
    def add_attempt(self, diagram: str):
        """Add a previous attempt to the history"""
        self.previous_attempts.append(diagram)
        self.attempt_number += 1


class RegenerationLogger:
    """Specialized logger for regeneration events"""
    
    def __init__(self, name: str = "mermaid_regeneration"):
        self.logger = logging.getLogger(name)
    
    def log_validation_failure(self, diagram: str, error: ValidationErrorContext):
        """Log validation failure with context"""
        self.logger.warning(
            f"🔄 MERMAID REGENERATION: Validation failed - {error.error_type.value} error: {error.error_message}"
        )
        if error.line_number:
            self.logger.debug(f"Error at line {error.line_number}")
        if error.suggested_fix:
            self.logger.debug(f"Suggested fix: {error.suggested_fix}")
    
    def log_regeneration_attempt(self, request: RegenerationRequest):
        """Log regeneration attempt"""
        self.logger.info(
            f"🔄 MERMAID REGENERATION: Attempt {request.attempt_number} for {request.validation_error.error_type.value} error"
        )
    
    def log_regeneration_success(self, result: RegenerationResult):
        """Log successful regeneration"""
        self.logger.info(
            f"✅ MERMAID REGENERATION: Success after {result.attempts_made} attempts in {result.regeneration_time:.2f}s"
        )
    
    def log_regeneration_failure(self, result: RegenerationResult):
        """Log regeneration failure"""
        self.logger.error(
            f"❌ MERMAID REGENERATION: Failed after {result.attempts_made} attempts. Errors: {'; '.join(result.errors_encountered)}"
        )
        if result.fallback_used:
            self.logger.info(f"🔄 MERMAID REGENERATION: Using fallback content")
    
    def log_performance_metrics(self, result: RegenerationResult):
        """Log performance metrics"""
        self.logger.debug(
            f"📊 MERMAID REGENERATION: Performance - {result.attempts_made} attempts, "
            f"{result.regeneration_time:.2f}s total, "
            f"{result.regeneration_time/max(result.attempts_made, 1):.2f}s per attempt"
        )


# Global configuration instance
_config_instance = None

def get_regeneration_config() -> RegenerationConfig:
    """Get singleton regeneration configuration"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RegenerationConfig.from_env()
    return _config_instance


# Global logger instance
_logger_instance = None

def get_regeneration_logger() -> RegenerationLogger:
    """Get singleton regeneration logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RegenerationLogger()
    return _logger_instance


class RegenerationPromptBuilder:
    """Builds context-rich correction prompts for LLM regeneration"""
    
    def __init__(self, config: Optional[RegenerationConfig] = None):
        self.config = config or get_regeneration_config()
        self.logger = get_regeneration_logger()
    
    def build_correction_prompt(
        self,
        original_query: str,
        invalid_diagram: str,
        validation_error: ValidationErrorContext,
        attempt_number: int = 1
    ) -> str:
        """
        Build comprehensive correction prompt with error context
        
        Args:
            original_query: The user's original request
            invalid_diagram: The diagram that failed validation
            validation_error: Structured error information
            attempt_number: Current attempt number
            
        Returns:
            Formatted correction prompt for LLM
        """
        prompt_parts = []
        
        # Header with context
        prompt_parts.append(
            "🔧 **MERMAID DIAGRAM CORRECTION REQUIRED**\n"
            f"The generated Mermaid diagram contains {validation_error.error_type.value} errors and cannot be rendered.\n"
        )
        
        # Original query context
        prompt_parts.append(f"**ORIGINAL USER REQUEST:**\n{original_query}\n")
        
        # Invalid diagram
        prompt_parts.append(f"**INVALID DIAGRAM CODE:**\n```mermaid\n{invalid_diagram}\n```\n")
        
        # Specific validation error
        prompt_parts.append(f"**VALIDATION ERROR:**\n{validation_error.error_message}\n")
        
        # Error-specific guidance
        if validation_error.suggested_fix:
            prompt_parts.append(f"**SUGGESTED FIX:**\n{validation_error.suggested_fix}\n")
        
        # Add targeted guidance based on error type
        guidance = self._get_error_specific_guidance(validation_error)
        if guidance:
            prompt_parts.append(f"**CORRECTION GUIDANCE:**\n{guidance}\n")
        
        # Include syntax rules if configured
        if self.config.include_syntax_rules:
            syntax_rules = self._get_mermaid_syntax_rules(validation_error.error_type)
            prompt_parts.append(f"**MERMAID SYNTAX RULES:**\n{syntax_rules}\n")
        
        # Include examples if configured
        if self.config.include_examples:
            examples = self._get_correction_examples(validation_error.error_type)
            if examples:
                prompt_parts.append(f"**CORRECTION EXAMPLES:**\n{examples}\n")
        
        # Attempt-specific instructions
        if attempt_number > 1:
            prompt_parts.append(
                f"**NOTE:** This is attempt #{attempt_number}. "
                "Please ensure the correction addresses the specific error mentioned above.\n"
            )
        
        # Final instructions
        prompt_parts.append(
            "**INSTRUCTIONS:**\n"
            "1. Fix the specific validation error mentioned above\n"
            "2. Ensure the diagram still accurately represents the original request\n"
            "3. Follow all Mermaid syntax rules exactly\n"
            "4. Return ONLY the corrected Mermaid diagram code\n"
            "5. Do not include explanations or markdown code blocks\n\n"
            "**CORRECTED DIAGRAM:**"
        )
        
        return "\n".join(prompt_parts)
    
    def _get_error_specific_guidance(self, error_context: ValidationErrorContext) -> str:
        """Get targeted guidance based on specific validation error"""
        error_type = error_context.error_type
        error_message = error_context.error_message.lower()
        
        guidance_map = {
            ErrorType.SYNTAX: self._get_syntax_guidance(error_message),
            ErrorType.STRUCTURE: self._get_structure_guidance(error_message),
            ErrorType.SEMANTIC: self._get_semantic_guidance(error_message),
            ErrorType.FORMAT: self._get_format_guidance(error_message),
            ErrorType.UNKNOWN: "Review the error message carefully and apply appropriate Mermaid syntax corrections."
        }
        
        return guidance_map.get(error_type, "")
    
    def _get_syntax_guidance(self, error_message: str) -> str:
        """Get guidance for syntax errors"""
        if 'uppercase' in error_message:
            return (
                "• Direction must be UPPERCASE: use 'TD', 'LR', 'TB', 'RL', or 'BT'\n"
                "• Example: 'graph TD' not 'graph td'"
            )
        elif 'parse error' in error_message:
            return (
                "• Check for missing semicolons, invalid characters, or malformed syntax\n"
                "• Ensure all arrows use proper syntax: '-->', '---', '-.->'\n"
                "• Verify node names contain only letters, numbers, and underscores"
            )
        else:
            return (
                "• Check basic syntax: graph declaration, node definitions, arrow connections\n"
                "• Ensure proper character escaping in labels"
            )
    
    def _get_structure_guidance(self, error_message: str) -> str:
        """Get guidance for structure errors"""
        if 'missing end' in error_message or 'unbalanced' in error_message:
            return (
                "• Every 'subgraph' must have a matching 'end' statement\n"
                "• Subgraphs must be properly nested\n"
                "• Example:\n"
                "  subgraph Main\n"
                "    A --> B\n"
                "  end"
            )
        elif 'graph declaration' in error_message:
            return (
                "• Every Mermaid diagram must start with a graph declaration\n"
                "• Use: 'graph TD', 'graph LR', 'flowchart TD', etc.\n"
                "• The declaration must be the first non-comment line"
            )
        else:
            return (
                "• Ensure proper diagram structure with correct declarations\n"
                "• Check that all structural elements are properly balanced"
            )
    
    def _get_semantic_guidance(self, error_message: str) -> str:
        """Get guidance for semantic errors"""
        return (
            "• Ensure all node references are defined\n"
            "• Check for circular dependencies in the diagram flow\n"
            "• Verify that relationships make logical sense"
        )
    
    def _get_format_guidance(self, error_message: str) -> str:
        """Get guidance for format errors"""
        if 'ampersand' in error_message:
            return (
                "• Escape ampersands in labels: use '&amp;' instead of '&'\n"
                "• Example: A[\"Data &amp; Processing\"] --> B"
            )
        elif 'quotes' in error_message or 'parentheses' in error_message:
            return (
                "• Quote labels containing special characters\n"
                "• Example: A[\"Process (Step 1)\"] --> B[\"Result\"]\n"
                "• Use double quotes for labels with spaces or special chars"
            )
        else:
            return (
                "• Check label formatting and character escaping\n"
                "• Ensure proper quoting of complex labels"
            )
    
    def _get_mermaid_syntax_rules(self, error_type: ErrorType) -> str:
        """Get relevant Mermaid syntax rules based on error type"""
        base_rules = (
            "• Start with graph declaration: 'graph TD', 'graph LR', etc.\n"
            "• Node syntax: NodeID[Label] or NodeID(Label) or NodeID{Label}\n"
            "• Arrow syntax: A --> B, A --- B, A -.-> B\n"
            "• Direction must be UPPERCASE: TD, LR, TB, RL, BT"
        )
        
        type_specific_rules = {
            ErrorType.SYNTAX: (
                "\n• Use only letters, numbers, underscores in node IDs\n"
                "• Escape special characters in labels with quotes\n"
                "• Use '&amp;' for ampersands in labels"
            ),
            ErrorType.STRUCTURE: (
                "\n• Subgraph syntax: 'subgraph Name' ... 'end'\n"
                "• All subgraphs must be properly closed\n"
                "• Graph declaration must be first line"
            ),
            ErrorType.FORMAT: (
                "\n• Quote labels with spaces: A[\"My Label\"]\n"
                "• Escape HTML entities: &amp;, &lt;, &gt;\n"
                "• Use proper character encoding"
            )
        }
        
        return base_rules + type_specific_rules.get(error_type, "")
    
    def _get_correction_examples(self, error_type: ErrorType) -> Optional[str]:
        """Get correction examples based on error type"""
        examples = {
            ErrorType.SYNTAX: (
                "❌ WRONG: graph td\n"
                "✅ CORRECT: graph TD\n\n"
                "❌ WRONG: A[Data & Processing] --> B\n"
                "✅ CORRECT: A[\"Data &amp; Processing\"] --> B"
            ),
            ErrorType.STRUCTURE: (
                "❌ WRONG:\n"
                "graph TD\n"
                "subgraph Main\n"
                "A --> B\n\n"
                "✅ CORRECT:\n"
                "graph TD\n"
                "subgraph Main\n"
                "A --> B\n"
                "end"
            ),
            ErrorType.FORMAT: (
                "❌ WRONG: A[Process (Step 1)] --> B\n"
                "✅ CORRECT: A[\"Process (Step 1)\"] --> B\n\n"
                "❌ WRONG: A[Data & Info] --> B\n"
                "✅ CORRECT: A[\"Data &amp; Info\"] --> B"
            )
        }
        
        return examples.get(error_type)


class MermaidRegenerator:
    """Core orchestrator for Mermaid diagram regeneration"""
    
    def __init__(self, llm_provider=None, config: Optional[RegenerationConfig] = None):
        self.llm_provider = llm_provider
        self.config = config or get_regeneration_config()
        self.prompt_builder = RegenerationPromptBuilder(self.config)
        self.logger = get_regeneration_logger()
    
    async def regenerate_diagram(
        self,
        invalid_diagram: str,
        validation_error: str,
        original_query: str,
        max_attempts: Optional[int] = None
    ) -> RegenerationResult:
        """
        Main regeneration orchestration method
        
        Args:
            invalid_diagram: The diagram that failed validation
            validation_error: The validation error message
            original_query: The original user query for context
            max_attempts: Maximum regeneration attempts (overrides config)
            
        Returns:
            RegenerationResult with outcome and details
        """
        start_time = time.time()
        max_attempts = max_attempts or self.config.max_attempts
        
        # Check if regeneration is enabled
        if not self.config.enabled:
            return RegenerationResult(
                status=RegenerationStatus.DISABLED,
                user_message="Diagram regeneration is disabled. Using original diagram."
            )
        
        # Create error context
        error_context = ValidationErrorContext.from_validation_error(validation_error)
        self.logger.log_validation_failure(invalid_diagram, error_context)
        
        # Initialize result
        result = RegenerationResult(status=RegenerationStatus.FAILED)
        
        # Create regeneration request
        request = RegenerationRequest(
            original_query=original_query,
            invalid_diagram=invalid_diagram,
            validation_error=error_context
        )
        
        # Attempt regeneration
        for attempt in range(1, max_attempts + 1):
            try:
                request.attempt_number = attempt
                self.logger.log_regeneration_attempt(request)
                
                # Check if we should attempt regeneration for this error
                if not self._should_attempt_regeneration(error_context, attempt):
                    self.logger.logger.info(f"Skipping regeneration attempt {attempt} for error type {error_context.error_type.value}")
                    break
                
                # Generate correction prompt
                correction_prompt = self.prompt_builder.build_correction_prompt(
                    original_query=original_query,
                    invalid_diagram=invalid_diagram,
                    validation_error=error_context,
                    attempt_number=attempt
                )
                
                # Get corrected diagram from LLM
                corrected_diagram = await self._get_corrected_diagram_from_llm(correction_prompt)
                
                if corrected_diagram:
                    # Validate the corrected diagram
                    is_valid, new_error = await self._validate_corrected_diagram(corrected_diagram)
                    
                    if is_valid:
                        # Success!
                        result.status = RegenerationStatus.SUCCESS
                        result.final_diagram = corrected_diagram
                        result.attempts_made = attempt
                        result.regeneration_time = time.time() - start_time
                        result.user_message = f"✨ Diagram automatically corrected (fixed {error_context.error_type.value} error)"
                        
                        self.logger.log_regeneration_success(result)
                        return result
                    else:
                        # Still invalid, add to attempts and continue
                        request.add_attempt(corrected_diagram)
                        result.add_error(f"Attempt {attempt}: {new_error}")
                        
                        # Update error context for next attempt
                        error_context = ValidationErrorContext.from_validation_error(new_error)
                else:
                    result.add_error(f"Attempt {attempt}: Failed to get response from LLM")
                
            except Exception as e:
                result.add_error(f"Attempt {attempt}: {str(e)}")
                self.logger.logger.error(f"Regeneration attempt {attempt} failed: {e}")
        
        # All attempts failed
        result.attempts_made = max_attempts
        result.regeneration_time = time.time() - start_time
        
        # Set up fallback if configured
        if self.config.fallback_to_text:
            fallback_content = self._generate_fallback_content(original_query, error_context)
            result.set_fallback(
                fallback_content,
                f"⚠️ Could not generate valid diagram after {max_attempts} attempts. Providing text description instead."
            )
        else:
            result.user_message = f"❌ Could not correct diagram after {max_attempts} attempts. Please check the syntax manually."
        
        self.logger.log_regeneration_failure(result)
        self.logger.log_performance_metrics(result)
        
        return result
    
    def _should_attempt_regeneration(self, error_context: ValidationErrorContext, attempt: int) -> bool:
        """Determine if regeneration should be attempted for this error"""
        # Don't attempt regeneration for unknown errors on later attempts
        if error_context.error_type == ErrorType.UNKNOWN and attempt > 1:
            return False
        
        # Always attempt for syntax and structure errors
        if error_context.error_type in [ErrorType.SYNTAX, ErrorType.STRUCTURE, ErrorType.FORMAT]:
            return True
        
        # Be more conservative with semantic errors
        if error_context.error_type == ErrorType.SEMANTIC and attempt > 1:
            return False
        
        return True
    
    async def _get_corrected_diagram_from_llm(self, correction_prompt: str) -> Optional[str]:
        """Get corrected diagram from LLM provider"""
        if not self.llm_provider:
            self.logger.logger.error("No LLM provider available for regeneration")
            return None
        
        try:
            # Use the LLM provider to get correction
            response = await self._call_llm_provider(correction_prompt)
            
            if response:
                # Extract diagram from response
                extracted_diagram = self._extract_diagram_from_response(response)
                return extracted_diagram
            
        except Exception as e:
            self.logger.logger.error(f"LLM provider call failed: {e}")
        
        return None
    
    async def _call_llm_provider(self, prompt: str) -> Optional[str]:
        """Call the LLM provider with the correction prompt"""
        try:
            # This will be integrated with the existing LLM provider system
            # For now, we'll use a placeholder that integrates with the existing agent
            if hasattr(self.llm_provider, 'generate_response'):
                response = await self.llm_provider.generate_response(prompt)
                return response
            elif hasattr(self.llm_provider, 'chat'):
                response = await self.llm_provider.chat(prompt)
                return response
            else:
                self.logger.logger.error("LLM provider does not have expected methods")
                return None
                
        except Exception as e:
            self.logger.logger.error(f"Error calling LLM provider: {e}")
            return None
    
    def _extract_diagram_from_response(self, llm_response: str) -> Optional[str]:
        """Extract mermaid diagram from LLM response"""
        if not llm_response:
            return None
        
        # Try to extract from markdown code blocks first
        mermaid_block_pattern = r'```mermaid\s*\n(.*?)\n```'
        match = re.search(mermaid_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try to extract from generic code blocks
        code_block_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(code_block_pattern, llm_response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Check if it looks like mermaid
            if content.startswith(('graph ', 'flowchart ', 'sequenceDiagram', 'classDiagram')):
                return content
        
        # Look for lines that start with graph/flowchart
        lines = llm_response.split('\n')
        diagram_lines = []
        in_diagram = False
        
        for line in lines:
            line = line.strip()
            if line.startswith(('graph ', 'flowchart ', 'sequenceDiagram', 'classDiagram')):
                in_diagram = True
                diagram_lines = [line]
            elif in_diagram:
                if line and not line.startswith(('**', '#', '>', '-', '*')):
                    diagram_lines.append(line)
                elif line == '' and diagram_lines:
                    # Empty line might be part of diagram
                    diagram_lines.append(line)
                else:
                    # Looks like end of diagram
                    break
        
        if diagram_lines:
            return '\n'.join(diagram_lines).strip()
        
        # Last resort: return the whole response if it looks like mermaid
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith(('graph ', 'flowchart ', 'sequenceDiagram', 'classDiagram')):
            return cleaned_response
        
        return None
    
    async def _validate_corrected_diagram(self, diagram: str) -> Tuple[bool, Optional[str]]:
        """Validate the corrected diagram"""
        try:
            # Import here to avoid circular imports
            from mermaid_validator import get_mermaid_validator
            validator = get_mermaid_validator()
            
            is_valid, error = validator.validate_mermaid_runtime(diagram)
            return is_valid, error
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _generate_fallback_content(self, original_query: str, error_context: ValidationErrorContext) -> str:
        """Generate fallback text content when diagram generation fails"""
        return (
            f"**System Architecture Overview**\n\n"
            f"Based on your request: \"{original_query}\"\n\n"
            f"*Note: A visual diagram could not be generated due to {error_context.error_type.value} errors. "
            f"The system attempted automatic correction but was unable to produce a valid diagram.*\n\n"
            f"**Error Details:** {error_context.error_message}\n\n"
            f"Please consider rephrasing your request or providing more specific details for diagram generation."
        )


# Global regenerator instance
_regenerator_instance = None

def get_mermaid_regenerator(llm_provider=None) -> MermaidRegenerator:
    """Get singleton Mermaid regenerator instance"""
    global _regenerator_instance
    if _regenerator_instance is None:
        _regenerator_instance = MermaidRegenerator(llm_provider)
    return _regenerator_instance