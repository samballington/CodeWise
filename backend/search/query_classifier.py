"""
Query Intent Classifier for Dynamic Search

Phase 3.1: Replaces static 50/50 hybrid search weighting with intelligent
query analysis that adapts search strategy based on query intent and characteristics.

This module implements multi-signal intent classification to determine optimal
vector vs BM25 weighting for each query dynamically.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of query intents for optimal search strategy selection."""
    SPECIFIC_SYMBOL = "specific_symbol"      # "authenticate_user function"
    CONCEPTUAL = "conceptual"                # "how does authentication work"
    STRUCTURAL = "structural"                # "show me the class hierarchy"
    EXPLORATORY = "exploratory"             # "what handles user registration"
    DEBUGGING = "debugging"                  # "error in login validation"


@dataclass
class QueryAnalysis:
    """Complete analysis of query with intent classification and search weights."""
    intent: QueryIntent
    confidence: float
    vector_weight: float      # 0.0 - 1.0
    bm25_weight: float        # 0.0 - 1.0
    signals: Dict[str, float] # Individual signal scores
    reasoning: List[str]      # Human-readable reasoning for weight selection
    detected_symbols: List[str]  # Programming symbols detected in query
    technical_terms: List[str]   # Technical programming terms found


class QueryClassifier:
    """
    Multi-signal query intent classifier for dynamic search optimization.
    
    Analyzes queries using linguistic patterns, technical indicators, and 
    contextual signals to determine optimal search strategy weighting.
    
    Architecture:
    - Signal-based classification (syntax, semantics, context)
    - Intent-driven weight calculation
    - Confidence scoring for adaptive behavior
    - Explainable reasoning for debugging
    """
    
    def __init__(self):
        """Initialize classifier with pattern libraries and signal extractors."""
        
        # Technical programming terms (favor BM25 for exact matches)
        self.technical_terms = {
            # Language constructs
            'function', 'class', 'method', 'variable', 'constant', 'interface',
            'type', 'enum', 'struct', 'module', 'package', 'namespace',
            'async', 'await', 'return', 'yield', 'import', 'export', 'from',
            
            # Access modifiers
            'public', 'private', 'protected', 'static', 'abstract', 'final',
            'const', 'let', 'var', 'def', 'lambda', 'arrow',
            
            # OOP concepts
            'extends', 'implements', 'inherits', 'override', 'super', 'self',
            'this', 'constructor', 'destructor', 'getter', 'setter',
            
            # Data structures
            'array', 'list', 'dict', 'map', 'set', 'tuple', 'object', 'json',
            'string', 'int', 'float', 'bool', 'char', 'byte',
            
            # Control flow
            'if', 'else', 'elif', 'switch', 'case', 'for', 'while', 'do',
            'break', 'continue', 'try', 'catch', 'except', 'finally', 'throw',
            
            # Common patterns
            'callback', 'handler', 'listener', 'decorator', 'annotation',
            'middleware', 'interceptor', 'factory', 'builder', 'singleton'
        }
        
        # Conceptual/natural language indicators (favor vector search)
        self.conceptual_terms = {
            'how', 'what', 'why', 'when', 'where', 'which', 'who',
            'explain', 'understand', 'concept', 'idea', 'approach', 'strategy',
            'pattern', 'best practice', 'example', 'tutorial', 'guide',
            'overview', 'summary', 'introduction', 'basics', 'fundamentals',
            'workflow', 'process', 'architecture', 'design', 'structure'
        }
        
        # Structural analysis terms (mixed approach)
        self.structural_terms = {
            'hierarchy', 'inheritance', 'dependency', 'relationship', 'connection',
            'structure', 'organization', 'architecture', 'design', 'pattern',
            'tree', 'graph', 'flow', 'diagram', 'schema', 'model'
        }
        
        # Debugging/problem-solving terms (favor BM25 for specific matches)
        self.debugging_terms = {
            'error', 'bug', 'issue', 'problem', 'fix', 'debug', 'troubleshoot',
            'exception', 'crash', 'fail', 'broken', 'not working', 'wrong',
            'unexpected', 'strange', 'weird', 'odd', 'trace', 'stack',
            'null', 'undefined', 'missing', 'empty', 'invalid'
        }
        
        # Symbol patterns (strongly favor BM25)
        self.symbol_patterns = [
            r'\b[a-z_][a-zA-Z0-9_]*\(\)',          # function_name()
            r'\b[A-Z][a-zA-Z0-9_]*\b',             # ClassName, CONSTANT
            r'\b[a-z_][a-zA-Z0-9_]*\.[a-z_]',      # object.method
            r'\b[a-z_][a-zA-Z0-9_]*::[a-z_]',      # namespace::function
            r'\b[a-z_][a-zA-Z0-9_]*\.[A-Z]',       # module.Class
            r'(?:def|function|class|interface|enum)\s+(\w+)',  # def function_name
            r'(?:const|let|var)\s+(\w+)',          # const variable_name
            r'(?:public|private|protected)\s+(\w+)',  # access modifier patterns
        ]
        
        # Question patterns (favor vector for natural language)
        self.question_patterns = [
            r'\bhow\s+(?:do|does|can|to|should)\b',
            r'\bwhat\s+(?:is|are|does|do|can)\b',
            r'\bwhy\s+(?:is|are|does|do)\b',
            r'\bwhere\s+(?:is|are|does|can)\b',
            r'\bwhen\s+(?:is|does|should|to)\b',
            r'\bwhich\s+(?:is|are|does|should)\b'
        ]
        
        # Code file extensions for context
        self.code_extensions = {
            '.py', '.js', '.ts', '.java', '.kt', '.go', '.rs', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.php', '.rb', '.swift', '.scala', '.clj',
            '.hs', '.ml', '.fs', '.dart', '.lua', '.r', '.m', '.pl'
        }
    
    def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryAnalysis:
        """
        Classify query intent and determine optimal search weights.
        
        Args:
            query: Search query string
            context: Optional context (recent files, current project, etc.)
            
        Returns:
            QueryAnalysis with intent, confidence, and optimal weights
        """
        query_lower = query.lower().strip()
        
        # Extract signals from query
        signals = self._extract_signals(query, query_lower, context)
        
        # Classify intent based on signals
        intent, intent_confidence = self._classify_intent(signals, query_lower)
        
        # Calculate optimal weights based on intent and signals
        vector_weight, bm25_weight = self._calculate_weights(intent, signals, intent_confidence)
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning(intent, signals, vector_weight, bm25_weight)
        
        # Extract detected programming elements
        detected_symbols = self._extract_symbols(query)
        technical_terms = self._extract_technical_terms(query_lower)
        
        return QueryAnalysis(
            intent=intent,
            confidence=intent_confidence,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            signals=signals,
            reasoning=reasoning,
            detected_symbols=detected_symbols,
            technical_terms=technical_terms
        )
    
    def _extract_signals(self, query: str, query_lower: str, context: Optional[Dict]) -> Dict[str, float]:
        """Extract and score various signals from the query."""
        
        signals = {}
        
        # 1. Technical Term Density
        technical_matches = sum(1 for term in self.technical_terms if term in query_lower)
        total_words = len(query_lower.split())
        signals['technical_density'] = technical_matches / max(total_words, 1)
        
        # 2. Conceptual Language Presence
        conceptual_matches = sum(1 for term in self.conceptual_terms if term in query_lower)
        signals['conceptual_density'] = conceptual_matches / max(total_words, 1)
        
        # 3. Symbol Pattern Detection
        symbol_matches = 0
        for pattern in self.symbol_patterns:
            symbol_matches += len(re.findall(pattern, query))
        signals['symbol_presence'] = min(symbol_matches / 3.0, 1.0)  # Normalize to 0-1
        
        # 4. Question Structure Detection
        question_score = 0
        for pattern in self.question_patterns:
            if re.search(pattern, query_lower):
                question_score += 1
        signals['question_structure'] = min(question_score / 2.0, 1.0)
        
        # 5. Query Length Analysis
        word_count = len(query.split())
        if word_count <= 3:
            signals['query_specificity'] = 0.8  # Short queries likely specific
        elif word_count <= 7:
            signals['query_specificity'] = 0.5  # Medium queries mixed
        else:
            signals['query_specificity'] = 0.2  # Long queries likely conceptual
        
        # 6. Punctuation and Syntax
        has_parens = '(' in query or ')' in query
        has_dots = '.' in query and not query.endswith('.')
        has_colons = '::' in query or '.' in query
        signals['code_syntax'] = (has_parens + has_dots + has_colons) / 3.0
        
        # 7. Debugging Language
        debug_matches = sum(1 for term in self.debugging_terms if term in query_lower)
        signals['debugging_context'] = min(debug_matches / 2.0, 1.0)
        
        # 8. Structural Analysis Terms
        structural_matches = sum(1 for term in self.structural_terms if term in query_lower)
        signals['structural_analysis'] = min(structural_matches / 2.0, 1.0)
        
        # 9. Context Signals (if provided)
        if context:
            # Recent file context
            if 'recent_files' in context:
                file_context_score = self._analyze_file_context(query_lower, context['recent_files'])
                signals['file_context_relevance'] = file_context_score
            
            # Project type context
            if 'project_type' in context:
                project_score = self._analyze_project_context(query_lower, context['project_type'])
                signals['project_context_relevance'] = project_score
        
        return signals
    
    def _classify_intent(self, signals: Dict[str, float], query_lower: str) -> Tuple[QueryIntent, float]:
        """Classify query intent based on extracted signals."""
        
        # Intent scoring based on signal combinations
        intent_scores = {
            QueryIntent.SPECIFIC_SYMBOL: (
                signals.get('symbol_presence', 0) * 0.4 +
                signals.get('technical_density', 0) * 0.3 +
                signals.get('code_syntax', 0) * 0.2 +
                signals.get('query_specificity', 0) * 0.1
            ),
            
            QueryIntent.CONCEPTUAL: (
                signals.get('conceptual_density', 0) * 0.4 +
                signals.get('question_structure', 0) * 0.3 +
                (1.0 - signals.get('query_specificity', 0)) * 0.2 +
                (1.0 - signals.get('technical_density', 0)) * 0.1
            ),
            
            QueryIntent.STRUCTURAL: (
                signals.get('structural_analysis', 0) * 0.5 +
                signals.get('technical_density', 0) * 0.2 +
                signals.get('conceptual_density', 0) * 0.2 +
                signals.get('question_structure', 0) * 0.1
            ),
            
            QueryIntent.DEBUGGING: (
                signals.get('debugging_context', 0) * 0.5 +
                signals.get('symbol_presence', 0) * 0.2 +
                signals.get('technical_density', 0) * 0.2 +
                signals.get('code_syntax', 0) * 0.1
            ),
            
            QueryIntent.EXPLORATORY: (
                signals.get('question_structure', 0) * 0.3 +
                signals.get('conceptual_density', 0) * 0.3 +
                (1.0 - signals.get('query_specificity', 0)) * 0.2 +
                signals.get('technical_density', 0) * 0.2
            )
        }
        
        # Find highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent, confidence = best_intent
        
        # Apply minimum confidence threshold
        if confidence < 0.3:
            # Default to exploratory for low confidence
            intent = QueryIntent.EXPLORATORY
            confidence = 0.3
        
        return intent, confidence
    
    def _calculate_weights(self, intent: QueryIntent, signals: Dict[str, float], 
                          confidence: float) -> Tuple[float, float]:
        """Calculate optimal vector and BM25 weights based on intent and signals."""
        
        # Base weight distributions by intent
        base_weights = {
            QueryIntent.SPECIFIC_SYMBOL: (0.2, 0.8),    # Heavily favor BM25 for exact matches
            QueryIntent.CONCEPTUAL: (0.8, 0.2),         # Heavily favor vector for concepts
            QueryIntent.STRUCTURAL: (0.6, 0.4),         # Moderate vector preference
            QueryIntent.DEBUGGING: (0.3, 0.7),          # Favor BM25 for specific debugging
            QueryIntent.EXPLORATORY: (0.7, 0.3)         # Favor vector for exploration
        }
        
        base_vector, base_bm25 = base_weights[intent]
        
        # Signal-based adjustments
        adjustments = 0.0
        
        # Strong symbol presence increases BM25 weight
        symbol_strength = signals.get('symbol_presence', 0)
        if symbol_strength > 0.5:
            adjustments -= (symbol_strength - 0.5) * 0.3  # Reduce vector weight
        
        # Strong conceptual language increases vector weight
        conceptual_strength = signals.get('conceptual_density', 0)
        if conceptual_strength > 0.3:
            adjustments += (conceptual_strength - 0.3) * 0.2  # Increase vector weight
        
        # Question structure favors vector search
        question_strength = signals.get('question_structure', 0)
        if question_strength > 0.5:
            adjustments += question_strength * 0.15
        
        # Technical density adjusts toward BM25
        technical_strength = signals.get('technical_density', 0)
        if technical_strength > 0.4:
            adjustments -= (technical_strength - 0.4) * 0.25
        
        # Apply confidence scaling (low confidence moves toward balanced)
        confidence_factor = max(confidence, 0.3)  # Minimum confidence
        balanced_pull = (1.0 - confidence_factor) * 0.3
        
        # Calculate final weights
        vector_weight = base_vector + adjustments - balanced_pull
        vector_weight = max(0.1, min(0.9, vector_weight))  # Clamp to reasonable range
        
        bm25_weight = 1.0 - vector_weight  # Ensure they sum to 1.0
        
        return vector_weight, bm25_weight
    
    def _generate_reasoning(self, intent: QueryIntent, signals: Dict[str, float],
                           vector_weight: float, bm25_weight: float) -> List[str]:
        """Generate human-readable reasoning for weight selection."""
        
        reasoning = []
        reasoning.append(f"Classified as {intent.value} query")
        
        # Explain major signal influences
        if signals.get('symbol_presence', 0) > 0.5:
            reasoning.append(f"Strong symbol patterns detected (score: {signals['symbol_presence']:.2f})")
        
        if signals.get('conceptual_density', 0) > 0.3:
            reasoning.append(f"Conceptual language present (score: {signals['conceptual_density']:.2f})")
        
        if signals.get('question_structure', 0) > 0.5:
            reasoning.append(f"Question structure detected (score: {signals['question_structure']:.2f})")
        
        if signals.get('technical_density', 0) > 0.4:
            reasoning.append(f"High technical term density (score: {signals['technical_density']:.2f})")
        
        # Explain weight selection
        if vector_weight > 0.6:
            reasoning.append("Favoring vector search for semantic understanding")
        elif bm25_weight > 0.6:
            reasoning.append("Favoring keyword search for exact matches")
        else:
            reasoning.append("Using balanced approach for mixed query characteristics")
        
        reasoning.append(f"Final weights: {vector_weight:.1f} vector / {bm25_weight:.1f} keyword")
        
        return reasoning
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract programming symbols from query."""
        symbols = []
        
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, query)
            symbols.extend(matches)
        
        # Remove duplicates and filter
        return list(set(symbol for symbol in symbols if len(symbol) > 1))
    
    def _extract_technical_terms(self, query_lower: str) -> List[str]:
        """Extract technical programming terms from query."""
        words = query_lower.split()
        return [word for word in words if word in self.technical_terms]
    
    def _analyze_file_context(self, query_lower: str, recent_files: List[str]) -> float:
        """Analyze relevance to recent files context."""
        if not recent_files:
            return 0.0
        
        # Check if query mentions file types or names present in recent files
        relevance_score = 0.0
        
        for file_path in recent_files:
            file_name = Path(file_path).name.lower()
            file_stem = Path(file_path).stem.lower()
            
            # Direct file name mention
            if file_stem in query_lower:
                relevance_score += 0.5
            
            # File extension context
            file_ext = Path(file_path).suffix
            if file_ext in self.code_extensions:
                ext_name = file_ext[1:]  # Remove dot
                if ext_name in query_lower:
                    relevance_score += 0.3
        
        return min(relevance_score, 1.0)
    
    def _analyze_project_context(self, query_lower: str, project_type: str) -> float:
        """Analyze relevance to project type context."""
        project_keywords = {
            'web': ['web', 'http', 'api', 'rest', 'server', 'client', 'frontend', 'backend'],
            'mobile': ['mobile', 'android', 'ios', 'app', 'ui', 'screen', 'view'],
            'data': ['data', 'database', 'sql', 'query', 'model', 'analytics', 'ml'],
            'game': ['game', 'engine', 'physics', 'render', 'scene', 'player', 'level'],
            'system': ['system', 'os', 'kernel', 'driver', 'memory', 'process', 'thread']
        }
        
        if project_type.lower() in project_keywords:
            keywords = project_keywords[project_type.lower()]
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            return min(matches / len(keywords), 1.0)
        
        return 0.0


# Convenience function for direct usage
def classify_query(query: str, context: Optional[Dict] = None) -> QueryAnalysis:
    """
    Classify a query and return analysis with optimal search weights.
    
    Args:
        query: Search query string
        context: Optional context information
        
    Returns:
        QueryAnalysis with intent classification and search weights
    """
    classifier = QueryClassifier()
    return classifier.classify_query(query, context)


if __name__ == "__main__":
    # CLI interface for testing the classifier
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Query Intent Classifier")
    parser.add_argument("query", help="Query to classify")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Test the classifier
    analysis = classify_query(args.query)
    
    print(f"Query: '{args.query}'")
    print(f"Intent: {analysis.intent.value}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Weights: {analysis.vector_weight:.2f} vector / {analysis.bm25_weight:.2f} keyword")
    
    if args.verbose:
        print("\nReasoning:")
        for reason in analysis.reasoning:
            print(f"  - {reason}")
        
        print(f"\nDetected Symbols: {analysis.detected_symbols}")
        print(f"Technical Terms: {analysis.technical_terms}")
        
        print(f"\nSignal Scores:")
        for signal, score in analysis.signals.items():
            print(f"  {signal}: {score:.3f}")