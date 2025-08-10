#!/usr/bin/env python3
"""
Response Formatting System for CodeWise - Task 5 Implementation
Provides standardized response templates, syntax highlighting, confidence scoring,
and structured metadata for the simplified 3-tool architecture.
"""

import re
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification for response templates"""
    AUTHENTICATION = "authentication"
    DATABASE_ENTITIES = "database_entities" 
    ARCHITECTURE = "architecture"
    IMPACT_ANALYSIS = "impact_analysis"
    GENERAL = "general"


class ConfidenceLevel(Enum):
    """Confidence levels for response quality assessment"""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.0-0.5


@dataclass
class CodeSnippet:
    """Structured code snippet with metadata"""
    content: str
    file_path: str
    language: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    highlighted: bool = False


@dataclass
class FileReference:
    """File reference with context"""
    file_path: str
    relevance_score: float
    context: str
    line_numbers: Optional[List[int]] = None


@dataclass
class ToolUsageMetadata:
    """Metadata about tool usage in response generation"""
    tool_name: str
    execution_time: float
    success: bool
    result_count: int
    confidence: float


@dataclass
class StandardizedResponse:
    """Standardized response format for all query types"""
    # Core content
    answer: str
    query_type: QueryType
    confidence_level: ConfidenceLevel
    confidence_score: float
    
    # Enhanced content
    code_snippets: List[CodeSnippet]
    file_references: List[FileReference]
    follow_up_suggestions: List[str]
    
    # Metadata
    tools_used: List[ToolUsageMetadata]
    response_time: float
    total_results_found: int
    
    # Context preservation
    context_summary: str
    contradictions_detected: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        
        # Convert enums to strings for JSON serialization
        result['query_type'] = self.query_type.value
        result['confidence_level'] = self.confidence_level.value
        
        return result


class ResponseFormatter:
    """Main response formatting system"""
    
    def __init__(self):
        self.query_classifiers = self._build_query_classifiers()
        self.response_templates = self._build_response_templates()
        self.syntax_highlighter = SyntaxHighlighter()
        self.confidence_assessor = ConfidenceAssessor()
        
        # Context tracking for contradiction detection
        self.tool_results_context = {}
    
    def format_response(self, 
                       raw_response: str,
                       tool_results: List[Dict[str, Any]],
                       query: str,
                       execution_metadata: Dict[str, Any]) -> StandardizedResponse:
        """
        Format a raw agent response into standardized format
        
        Args:
            raw_response: Raw text response from agent
            tool_results: Results from tool executions
            query: Original user query
            execution_metadata: Timing and execution data
            
        Returns:
            StandardizedResponse object
        """
        logger.info(f"🎨 FORMATTING RESPONSE for query: '{query[:50]}...'")
        
        # 1. Classify query type
        query_type = self._classify_query(query)
        logger.debug(f"Query classified as: {query_type.value}")
        
        # 2. Extract and format code snippets
        code_snippets = self._extract_code_snippets(raw_response, tool_results)
        
        # 3. Extract file references
        file_references = self._extract_file_references(tool_results)
        
        # 4. Assess confidence
        confidence_score, confidence_level = self.confidence_assessor.assess_confidence(
            raw_response, tool_results, query_type
        )
        
        # 5. Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(query_type, tool_results)
        
        # 6. Create tool usage metadata
        tools_metadata = self._create_tool_metadata(tool_results, execution_metadata)
        
        # 7. Detect contradictions
        contradictions = self._detect_contradictions(tool_results)
        
        # 8. Apply response template
        formatted_answer = self._apply_response_template(
            raw_response, query_type, code_snippets, file_references
        )
        
        # 9. Create context summary
        context_summary = self._create_context_summary(tool_results)
        
        response = StandardizedResponse(
            answer=formatted_answer,
            query_type=query_type,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            code_snippets=code_snippets,
            file_references=file_references,
            follow_up_suggestions=follow_up_suggestions,
            tools_used=tools_metadata,
            response_time=execution_metadata.get('total_time', 0.0),
            total_results_found=sum(len(tr.get('results', [])) for tr in tool_results),
            context_summary=context_summary,
            contradictions_detected=contradictions
        )
        
        logger.info(f"✅ Response formatted: {confidence_level.value} confidence, "
                   f"{len(code_snippets)} snippets, {len(file_references)} files")
        
        return response 
   
    def _build_query_classifiers(self) -> Dict[QueryType, List[str]]:
        """Build query classification patterns"""
        return {
            QueryType.AUTHENTICATION: [
                'authentication', 'auth', 'login', 'signin', 'signup', 'password',
                'session', 'token', 'jwt', 'oauth', 'security', 'user login'
            ],
            QueryType.DATABASE_ENTITIES: [
                'database', 'entities', 'models', 'schema', 'table', 'orm',
                'entity', 'model', 'db', 'data model', 'database schema'
            ],
            QueryType.ARCHITECTURE: [
                'architecture', 'system', 'overview', 'structure', 'design',
                'components', 'modules', 'flow', 'how does', 'explain'
            ],
            QueryType.IMPACT_ANALYSIS: [
                'what breaks', 'impact', 'change', 'affect', 'dependencies',
                'what happens if', 'break', 'modify', 'update'
            ]
        }
    
    def _build_response_templates(self) -> Dict[QueryType, str]:
        """Build response templates for different query types"""
        return {
            QueryType.AUTHENTICATION: """
## Authentication System Analysis

{answer}

### Key Components
{code_snippets}

### Related Files
{file_references}

### Security Considerations
- Authentication flow involves {file_count} files
- Key security patterns identified
- Dependencies mapped for impact analysis
""",
            QueryType.DATABASE_ENTITIES: """
## Database Entities Overview

{answer}

### Entity Definitions
{code_snippets}

### Entity Relationships
{file_references}

### Schema Summary
- {entity_count} entities discovered
- Relationships mapped across {file_count} files
- ORM patterns identified
""",
            QueryType.ARCHITECTURE: """
## System Architecture Analysis

{answer}

### Core Components
{code_snippets}

### Architecture Files
{file_references}

### System Overview
- {component_count} key components identified
- Architecture spans {file_count} files
- Design patterns documented
""",
            QueryType.IMPACT_ANALYSIS: """
## Impact Analysis Report

{answer}

### Affected Code
{code_snippets}

### Dependency Chain
{file_references}

### Impact Summary
- {affected_files} files would be affected
- Risk level: {risk_level}
- Recommended testing approach included
""",
            QueryType.GENERAL: """
## Analysis Results

{answer}

### Relevant Code
{code_snippets}

### Related Files
{file_references}
"""
        }
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type based on keywords"""
        query_lower = query.lower()
        
        # Score each query type
        scores = {}
        for query_type, keywords in self.query_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[query_type] = score
        
        # Return highest scoring type, default to GENERAL
        if scores:
            return max(scores, key=scores.get)
        return QueryType.GENERAL
    
    def _extract_code_snippets(self, raw_response: str, tool_results: List[Dict]) -> List[CodeSnippet]:
        """Extract and format code snippets from response and tool results"""
        snippets = []
        
        # Extract from tool results first (more reliable)
        for tool_result in tool_results:
            if tool_result.get('tool_name') == 'examine_files':
                snippets.extend(self._extract_snippets_from_examine_files(tool_result))
            elif tool_result.get('tool_name') == 'smart_search':
                snippets.extend(self._extract_snippets_from_search(tool_result))
        
        # Extract from raw response as fallback
        response_snippets = self._extract_snippets_from_text(raw_response)
        snippets.extend(response_snippets)
        
        # Remove duplicates and apply syntax highlighting
        unique_snippets = self._deduplicate_snippets(snippets)
        for snippet in unique_snippets:
            if not snippet.highlighted:
                snippet.content = self.syntax_highlighter.highlight(
                    snippet.content, snippet.language
                )
                snippet.highlighted = True
        
        return unique_snippets[:10]  # Limit to 10 snippets
    
    def _extract_snippets_from_examine_files(self, tool_result: Dict) -> List[CodeSnippet]:
        """Extract code snippets from examine_files tool results"""
        snippets = []
        result_text = tool_result.get('result', '')
        
        # Look for code blocks in examine_files output
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, result_text, re.DOTALL)
        
        for language, content in matches:
            # Try to extract file path from context
            file_path = self._extract_file_path_from_context(result_text, content)
            
            snippets.append(CodeSnippet(
                content=content.strip(),
                file_path=file_path or "unknown",
                language=language or "text",
                highlighted=False
            ))
        
        return snippets
    
    def _extract_snippets_from_search(self, tool_result: Dict) -> List[CodeSnippet]:
        """Extract code snippets from smart_search results"""
        snippets = []
        result_text = tool_result.get('result', '')
        
        # Look for search result snippets
        snippet_pattern = r'📝 Code Snippet:\s*\n(.*?)(?=\n   ─|$)'
        matches = re.findall(snippet_pattern, result_text, re.DOTALL)
        
        for content in matches:
            # Clean up the content
            cleaned_content = re.sub(r'^\s*', '', content, flags=re.MULTILINE)
            
            # Try to detect language from content
            language = self._detect_language_from_content(cleaned_content)
            
            snippets.append(CodeSnippet(
                content=cleaned_content.strip(),
                file_path="search_result",
                language=language,
                highlighted=False
            ))
        
        return snippets
    
    def _extract_snippets_from_text(self, text: str) -> List[CodeSnippet]:
        """Extract code snippets from raw text"""
        snippets = []
        
        # Look for existing code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        for language, content in matches:
            snippets.append(CodeSnippet(
                content=content.strip(),
                file_path="response_text",
                language=language or "text",
                highlighted=False
            ))
        
        return snippets    

    def _extract_file_references(self, tool_results: List[Dict]) -> List[FileReference]:
        """Extract file references from tool results"""
        references = []
        
        for tool_result in tool_results:
            tool_name = tool_result.get('tool_name', '')
            result_text = tool_result.get('result', '')
            
            if tool_name == 'smart_search':
                references.extend(self._extract_references_from_search(result_text))
            elif tool_name == 'examine_files':
                references.extend(self._extract_references_from_examine(result_text))
            elif tool_name == 'analyze_relationships':
                references.extend(self._extract_references_from_relationships(result_text))
        
        # Remove duplicates and sort by relevance
        unique_refs = {}
        for ref in references:
            key = ref.file_path
            if key not in unique_refs or ref.relevance_score > unique_refs[key].relevance_score:
                unique_refs[key] = ref
        
        sorted_refs = sorted(unique_refs.values(), key=lambda x: x.relevance_score, reverse=True)
        return sorted_refs[:8]  # Limit to 8 references
    
    def _extract_references_from_search(self, result_text: str) -> List[FileReference]:
        """Extract file references from search results"""
        references = []
        
        # Legacy pattern: 📄 RESULT #n: path  🎯 Relevance: TIER (score)
        legacy_pattern = r'📄 RESULT #\d+: ([^\n]+)\s+🎯 Relevance: [🟢🟡🔴] \w+ \(([0-9.]+)\)'
        for file_path, relevance_str in re.findall(legacy_pattern, result_text):
            try:
                references.append(FileReference(
                    file_path=file_path.strip(),
                    relevance_score=float(relevance_str),
                    context="Found in search results"
                ))
            except ValueError:
                pass

        # New context-aware format: **1. path** (score: 0.800, strategy: ...)
        modern_pattern = r"\*\*\s*\d+\.\s*([^*\n]+)\*\*\s*\(score:\s*([0-9.]+)"
        for file_path, score_str in re.findall(modern_pattern, result_text):
            # Normalize and record
            try:
                references.append(FileReference(
                    file_path=file_path.strip(),
                    relevance_score=float(score_str),
                    context="Found in search results"
                ))
            except ValueError:
                pass
        
        return references
    
    def _extract_references_from_examine(self, result_text: str) -> List[FileReference]:
        """Extract file references from examine_files results"""
        references = []
        
        # Pattern to match examined files
        file_pattern = r'📁 FILE: ([^\n]+)'
        matches = re.findall(file_pattern, result_text)
        
        for file_path in matches:
            references.append(FileReference(
                file_path=file_path.strip(),
                relevance_score=0.8,  # High relevance for examined files
                context="Examined in detail"
            ))
        
        # Maven POM enrichment: detect parent POM and dependencyManagement sections
        try:
            # Only proceed if we examined a pom.xml
            pom_files = [fp for fp in matches if fp.strip().lower().endswith('pom.xml')]
            if pom_files:
                # Extract XML blocks (best-effort) to inspect parent/dependencyManagement
                # Look for relativePath inside the examine output
                rel_match = re.search(r'<relativePath>\s*([^<]+)\s*</relativePath>', result_text)
                depmgmt_present = bool(re.search(r'<\s*dependencyManagement\b', result_text))

                # Add a reference for dependencyManagement section presence
                if depmgmt_present:
                    for pom in pom_files:
                        references.append(FileReference(
                            file_path=pom.strip(),
                            relevance_score=0.6,
                            context="dependencyManagement section present"
                        ))

                # Resolve parent POM file path if relativePath found (best-effort without FS IO)
                if rel_match:
                    parent_rel = rel_match.group(1).strip()
                    for pom in pom_files:
                        try:
                            pom_path = Path(pom.strip())
                            parent_path = (pom_path.parent / parent_rel).resolve().as_posix()
                            # Normalize to workspace-style relative if path includes /workspace already omitted
                            references.append(FileReference(
                                file_path=parent_path,
                                relevance_score=0.55,
                                context="Parent POM"
                            ))
                        except Exception:
                            # On any failure, skip adding parent path
                            pass
        except Exception:
            # Do not fail reference extraction for enrichment errors
            pass
        
        return references
    
    def _extract_references_from_relationships(self, result_text: str) -> List[FileReference]:
        """Extract file references from relationship analysis"""
        references = []
        
        # Pattern to match related files
        related_pattern = r'📤 USAGE.*?• ([^\n]+)'
        matches = re.findall(related_pattern, result_text, re.DOTALL)
        
        for file_path in matches:
            if file_path.strip() and not file_path.startswith('None'):
                references.append(FileReference(
                    file_path=file_path.strip(),
                    relevance_score=0.7,  # Medium-high relevance for related files
                    context="Related through dependencies"
                ))
        
        return references
    
    def _generate_follow_up_suggestions(self, query_type: QueryType, tool_results: List[Dict]) -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = []
        
        if query_type == QueryType.AUTHENTICATION:
            suggestions = [
                "How is user session management implemented?",
                "What security measures are in place for password handling?",
                "Show me the user registration flow",
                "How are authentication tokens validated?"
            ]
        elif query_type == QueryType.DATABASE_ENTITIES:
            suggestions = [
                "Show me the relationships between these entities",
                "How is data validation implemented?",
                "What would break if I modify the User entity?",
                "Explain the database migration strategy"
            ]
        elif query_type == QueryType.ARCHITECTURE:
            suggestions = [
                "How do these components communicate?",
                "What are the main entry points of the system?",
                "Show me the data flow through the application",
                "What external dependencies does this system have?"
            ]
        elif query_type == QueryType.IMPACT_ANALYSIS:
            suggestions = [
                "What tests should I run after making this change?",
                "Are there any breaking changes I should be aware of?",
                "How can I minimize the impact of this change?",
                "What other components depend on this?"
            ]
        else:  # GENERAL
            suggestions = [
                "Can you show me more details about this implementation?",
                "How does this relate to other parts of the system?",
                "What would be the best way to modify this code?",
                "Are there any similar patterns used elsewhere?"
            ]
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _create_tool_metadata(self, tool_results: List[Dict], execution_metadata: Dict) -> List[ToolUsageMetadata]:
        """Create metadata about tool usage"""
        metadata = []
        
        for tool_result in tool_results:
            tool_name = tool_result.get('tool_name', 'unknown')
            execution_time = tool_result.get('execution_time', 0.0)
            success = 'error' not in tool_result.get('result', '').lower()
            result_count = len(tool_result.get('results', []))
            
            # Calculate confidence based on result quality
            confidence = self._calculate_tool_confidence(tool_result)
            
            metadata.append(ToolUsageMetadata(
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                result_count=result_count,
                confidence=confidence
            ))
        
        return metadata
    
    def _calculate_tool_confidence(self, tool_result: Dict) -> float:
        """Calculate confidence score for a tool result"""
        result_text = tool_result.get('result', '')
        
        # Base confidence
        confidence = 0.5
        
        # Boost for successful results
        if 'error' not in result_text.lower() and 'failed' not in result_text.lower():
            confidence += 0.2
        
        # Boost for substantial content
        if len(result_text) > 500:
            confidence += 0.1
        
        # Boost for structured output
        if '📄' in result_text or '🔗' in result_text or '🧠' in result_text:
            confidence += 0.1
        
        # Penalty for empty or minimal results
        if len(result_text) < 100:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _detect_contradictions(self, tool_results: List[Dict]) -> List[str]:
        """Detect contradictions between tool results"""
        contradictions = []
        
        # Store current results for future contradiction detection
        for tool_result in tool_results:
            tool_name = tool_result.get('tool_name')
            result_text = tool_result.get('result', '')
            
            # Simple contradiction detection - look for conflicting statements
            if tool_name in self.tool_results_context:
                previous_result = self.tool_results_context[tool_name]
                
                # Check for obvious contradictions (simplified)
                if self._results_contradict(previous_result, result_text):
                    contradictions.append(
                        f"Contradiction detected in {tool_name} results"
                    )
            
            self.tool_results_context[tool_name] = result_text
        
        return contradictions
    
    def _results_contradict(self, previous: str, current: str) -> bool:
        """Simple contradiction detection between two results"""
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP techniques
        
        # Look for opposite statements
        contradiction_patterns = [
            (r'does not exist', r'exists'),
            (r'not found', r'found'),
            (r'no.*files', r'\d+.*files'),
            (r'empty', r'contains')
        ]
        
        for negative_pattern, positive_pattern in contradiction_patterns:
            if (re.search(negative_pattern, previous, re.IGNORECASE) and 
                re.search(positive_pattern, current, re.IGNORECASE)):
                return True
            if (re.search(positive_pattern, previous, re.IGNORECASE) and 
                re.search(negative_pattern, current, re.IGNORECASE)):
                return True
        
        return False
    
    def _apply_response_template(self, raw_response: str, query_type: QueryType, 
                                code_snippets: List[CodeSnippet], 
                                file_references: List[FileReference]) -> str:
        """Apply response template based on query type"""
        template = self.response_templates.get(query_type, self.response_templates[QueryType.GENERAL])
        
        # Format code snippets for template
        snippets_text = self._format_snippets_for_template(code_snippets)
        
        # Format file references for template
        references_text = self._format_references_for_template(file_references)
        
        # Calculate template variables
        template_vars = {
            'answer': raw_response,
            'code_snippets': snippets_text,
            'file_references': references_text,
            'file_count': len(file_references),
            'entity_count': self._count_entities_in_snippets(code_snippets),
            'component_count': self._count_components_in_snippets(code_snippets),
            'affected_files': len(file_references),
            'risk_level': self._assess_risk_level(file_references)
        }
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            return raw_response  # Fallback to raw response
    
    def _format_snippets_for_template(self, snippets: List[CodeSnippet]) -> str:
        """Format code snippets for template insertion"""
        if not snippets:
            return "*No code snippets available*"
        
        formatted = []
        for i, snippet in enumerate(snippets[:5], 1):  # Limit to 5 for template
            formatted.append(f"""
**{i}. {snippet.file_path}** ({snippet.language})
```{snippet.language}
{snippet.content}
```
""")
        
        return "\n".join(formatted)
    
    def _format_references_for_template(self, references: List[FileReference]) -> str:
        """Format file references for template insertion"""
        if not references:
            return "*No file references available*"
        
        formatted = []
        for ref in references[:8]:  # Limit to 8 for template
            relevance_indicator = "🟢" if ref.relevance_score > 0.7 else "🟡" if ref.relevance_score > 0.4 else "🔴"
            formatted.append(f"- {relevance_indicator} **{ref.file_path}** - {ref.context}")
        
        return "\n".join(formatted)
    
    def _count_entities_in_snippets(self, snippets: List[CodeSnippet]) -> int:
        """Count database entities in code snippets"""
        entity_patterns = [
            r'@Entity', r'class.*Model', r'CREATE TABLE', r'models\.Model',
            r'sequelize\.define', r'mongoose\.Schema'
        ]
        
        count = 0
        for snippet in snippets:
            for pattern in entity_patterns:
                count += len(re.findall(pattern, snippet.content, re.IGNORECASE))
        
        return count
    
    def _count_components_in_snippets(self, snippets: List[CodeSnippet]) -> int:
        """Count architectural components in code snippets"""
        component_patterns = [
            r'class.*Controller', r'class.*Service', r'class.*Repository',
            r'@Component', r'@Service', r'@Controller', r'function.*Component'
        ]
        
        count = 0
        for snippet in snippets:
            for pattern in component_patterns:
                count += len(re.findall(pattern, snippet.content, re.IGNORECASE))
        
        return max(count, len(snippets))  # At least as many as snippets
    
    def _assess_risk_level(self, file_references: List[FileReference]) -> str:
        """Assess risk level based on file references"""
        if len(file_references) > 10:
            return "HIGH"
        elif len(file_references) > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _create_context_summary(self, tool_results: List[Dict]) -> str:
        """Create a summary of the context used in response generation"""
        summary_parts = []
        
        for tool_result in tool_results:
            tool_name = tool_result.get('tool_name', 'unknown')
            result_text = tool_result.get('result', '')
            
            # Extract key information from each tool
            if tool_name == 'smart_search':
                results_count = result_text.count('📄 RESULT #')
                summary_parts.append(f"Search found {results_count} relevant results")
            elif tool_name == 'examine_files':
                files_count = result_text.count('📁 FILE:')
                summary_parts.append(f"Examined {files_count} files in detail")
            elif tool_name == 'analyze_relationships':
                if 'RELATIONSHIP ANALYSIS' in result_text:
                    summary_parts.append("Analyzed code relationships and dependencies")
        
        return "; ".join(summary_parts) if summary_parts else "No context available"
    
    def _extract_file_path_from_context(self, context: str, content: str) -> Optional[str]:
        """Extract file path from context around code content"""
        # Look for file path patterns in the context
        file_patterns = [
            r'📁 FILE: ([^\n]+)',
            r'FILE: ([^\n]+)',
            r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z]{2,4})'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, context)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _detect_language_from_content(self, content: str) -> str:
        """Detect programming language from code content"""
        # Simple language detection based on patterns
        if re.search(r'def\s+\w+|import\s+\w+|from\s+\w+', content):
            return "python"
        elif re.search(r'function\s+\w+|const\s+\w+|let\s+\w+', content):
            return "javascript"
        elif re.search(r'interface\s+\w+|type\s+\w+|export\s+', content):
            return "typescript"
        elif re.search(r'public\s+class|private\s+\w+|@\w+', content):
            return "java"
        else:
            return "text"
    
    def _deduplicate_snippets(self, snippets: List[CodeSnippet]) -> List[CodeSnippet]:
        """Remove duplicate code snippets"""
        seen = set()
        unique_snippets = []
        
        for snippet in snippets:
            # Create a hash based on content and file path
            snippet_hash = hash(f"{snippet.file_path}:{snippet.content[:100]}")
            
            if snippet_hash not in seen:
                seen.add(snippet_hash)
                unique_snippets.append(snippet)
        
        return unique_snippets


class SyntaxHighlighter:
    """Syntax highlighting for code snippets"""
    
    def highlight(self, code: str, language: str) -> str:
        """Apply syntax highlighting to code"""
        # For now, just wrap in markdown code blocks
        # In a full implementation, this could use pygments or similar
        return f"```{language}\n{code}\n```"


class ConfidenceAssessor:
    """Assess confidence in response quality"""
    
    def assess_confidence(self, response: str, tool_results: List[Dict], 
                         query_type: QueryType) -> Tuple[float, ConfidenceLevel]:
        """Assess confidence in response quality"""
        confidence_score = 0.5  # Base confidence
        
        # Factor 1: Response length and detail
        if len(response) > 500:
            confidence_score += 0.1
        if len(response) > 1000:
            confidence_score += 0.1
        
        # Factor 2: Tool success rate
        successful_tools = sum(1 for tr in tool_results 
                             if 'error' not in tr.get('result', '').lower())
        if tool_results:
            tool_success_rate = successful_tools / len(tool_results)
            confidence_score += tool_success_rate * 0.2
        
        # Factor 3: Number of results found
        total_results = sum(len(tr.get('results', [])) for tr in tool_results)
        if total_results > 5:
            confidence_score += 0.1
        if total_results > 10:
            confidence_score += 0.1
        
        # Factor 4: Query type specific factors
        if query_type in [QueryType.AUTHENTICATION, QueryType.DATABASE_ENTITIES]:
            # These require specific patterns
            if any(pattern in response.lower() for pattern in ['class', 'function', 'import']):
                confidence_score += 0.1
        
        # Normalize to 0-1 range
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine confidence level
        if confidence_score >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        return confidence_score, level