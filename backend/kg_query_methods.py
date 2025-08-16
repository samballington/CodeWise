"""
Knowledge Graph Query Methods for Tools

Phase 2.3.3: Adds new KG-specific methods to the tool interface for comprehensive
symbol neighborhood exploration, dependency analysis, and architectural insights.

These methods provide direct access to the Knowledge Graph's analytical power
through clean, tool-friendly interfaces that can be integrated into any backend.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

# Import Phase 2 Knowledge Graph components
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from storage.database_manager import DatabaseManager
except ImportError:
    # Docker environment - backend is working directory
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class QueryScope(Enum):
    """Scope for KG query operations."""
    SYMBOL = "symbol"
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"


class RelationshipDirection(Enum):
    """Direction for relationship traversal."""
    INCOMING = "incoming"  # What calls/uses this
    OUTGOING = "outgoing"  # What this calls/uses
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class SymbolInfo:
    """Rich symbol information from Knowledge Graph."""
    id: str
    name: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    signature: str = None
    docstring: str = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'file_path': self.file_path,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'signature': self.signature,
            'docstring': self.docstring,
            'metadata': self.metadata or {}
        }


@dataclass
class RelationshipInfo:
    """Rich relationship information from Knowledge Graph."""
    source: SymbolInfo
    target: SymbolInfo
    relationship_type: str
    direction: str
    depth: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source.to_dict(),
            'target': self.target.to_dict(),
            'relationship_type': self.relationship_type,
            'direction': self.direction,
            'depth': self.depth,
            'metadata': self.metadata or {}
        }


@dataclass
class DependencyAnalysis:
    """Comprehensive dependency analysis result."""
    target_symbol: SymbolInfo
    direct_dependencies: List[RelationshipInfo]
    transitive_dependencies: List[RelationshipInfo]
    dependent_symbols: List[RelationshipInfo]
    dependency_groups: Dict[str, List[RelationshipInfo]]
    circular_dependencies: List[List[str]]
    complexity_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'target_symbol': self.target_symbol.to_dict(),
            'direct_dependencies': [r.to_dict() for r in self.direct_dependencies],
            'transitive_dependencies': [r.to_dict() for r in self.transitive_dependencies],
            'dependent_symbols': [r.to_dict() for r in self.dependent_symbols],
            'dependency_groups': {
                k: [r.to_dict() for r in v] for k, v in self.dependency_groups.items()
            },
            'circular_dependencies': self.circular_dependencies,
            'complexity_metrics': self.complexity_metrics
        }


class KGQueryMethods:
    """
    Phase 2.3.3: Knowledge Graph Query Methods for Tools
    
    Provides powerful, tool-friendly KG query capabilities including:
    - Symbol neighborhood exploration
    - Dependency impact analysis
    - Architectural pattern detection
    - Call graph traversal
    - Inheritance hierarchy mapping
    - Cross-module relationship analysis
    
    Design Goals:
    - Clean, simple interfaces for tool integration
    - Rich, structured results with metadata
    - Performance-optimized recursive queries
    - Comprehensive relationship context
    """
    
    def __init__(self, db_path: str = "codewise.db"):
        """
        Initialize KG Query Methods.
        
        Args:
            db_path: Path to SQLite Knowledge Graph database
        """
        try:
            self.db_manager = DatabaseManager(db_path)
            self.kg_available = True
            logger.info("🔗 KG Query Methods initialized successfully")
        except Exception as e:
            logger.warning(f"KG unavailable: {e}. Query methods will return empty results.")
            self.kg_available = False
            self.db_manager = None
    
    # ==================== SYMBOL EXPLORATION METHODS ====================
    
    def find_symbol(self, symbol_name: str, exact_match: bool = True) -> List[SymbolInfo]:
        """
        Find symbols in the Knowledge Graph by name.
        
        Args:
            symbol_name: Name of symbol to find
            exact_match: Whether to use exact matching or fuzzy search
            
        Returns:
            List of matching symbols with full metadata
        """
        if not self.kg_available:
            return []
        
        try:
            nodes = self.db_manager.get_nodes_by_name(symbol_name, exact_match=exact_match)
            return [self._node_to_symbol_info(node) for node in nodes]
        except Exception as e:
            logger.error(f"Failed to find symbol '{symbol_name}': {e}")
            return []
    
    def get_symbol_by_location(self, file_path: str, line_number: int) -> Optional[SymbolInfo]:
        """
        Find symbol at specific file location.
        
        Args:
            file_path: Path to source file
            line_number: Line number in file
            
        Returns:
            Symbol at that location, if any
        """
        if not self.kg_available:
            return None
        
        try:
            # Get all symbols in file
            file_symbols = self.db_manager.get_nodes_by_file(file_path)
            
            # Find symbol containing the line number
            for node in file_symbols:
                line_start = node.get('line_start', 0)
                line_end = node.get('line_end', line_start)
                
                if line_start <= line_number <= line_end:
                    return self._node_to_symbol_info(node)
            
            return None
        except Exception as e:
            logger.error(f"Failed to find symbol at {file_path}:{line_number}: {e}")
            return None
    
    def explore_symbol_neighborhood(self, symbol_name: str, 
                                  max_depth: int = 3,
                                  include_siblings: bool = True) -> Dict[str, Any]:
        """
        Explore the complete 'neighborhood' of relationships around a symbol.
        
        Args:
            symbol_name: Symbol to explore
            max_depth: Maximum relationship traversal depth
            include_siblings: Whether to include symbols in same file
            
        Returns:
            Comprehensive neighborhood analysis with all relationship types
        """
        if not self.kg_available:
            return {'error': 'Knowledge Graph unavailable'}
        
        try:
            symbols = self.find_symbol(symbol_name, exact_match=True)
            if not symbols:
                symbols = self.find_symbol(symbol_name, exact_match=False)
            
            if not symbols:
                return {'error': f"Symbol '{symbol_name}' not found in Knowledge Graph"}
            
            neighborhoods = {}
            
            for symbol in symbols:
                neighborhood = self._analyze_symbol_neighborhood(symbol, max_depth, include_siblings)
                neighborhoods[symbol.id] = neighborhood
            
            return {
                'symbol_name': symbol_name,
                'neighborhoods': neighborhoods,
                'total_symbols_analyzed': len(symbols),
                'summary': self._generate_neighborhood_summary(neighborhoods)
            }
            
        except Exception as e:
            logger.error(f"Failed to explore neighborhood for '{symbol_name}': {e}")
            return {'error': str(e)}
    
    # ==================== RELATIONSHIP TRAVERSAL METHODS ====================
    
    def find_callers(self, symbol_name: str, max_depth: int = 3) -> List[RelationshipInfo]:
        """
        Find all symbols that call the target symbol.
        
        Args:
            symbol_name: Symbol to find callers for
            max_depth: Maximum call graph traversal depth
            
        Returns:
            List of caller relationships with full context
        """
        if not self.kg_available:
            return []
        
        try:
            symbols = self.find_symbol(symbol_name, exact_match=True)
            if not symbols:
                return []
            
            all_relationships = []
            
            for symbol in symbols:
                callers = self.db_manager.find_callers(symbol.id, max_depth)
                
                for caller_data in callers:
                    caller_symbol = self._node_to_symbol_info(caller_data)
                    relationship = RelationshipInfo(
                        source=caller_symbol,
                        target=symbol,
                        relationship_type='calls',
                        direction='incoming',
                        depth=caller_data.get('depth', 0),
                        metadata={'call_context': caller_data}
                    )
                    all_relationships.append(relationship)
            
            return all_relationships
            
        except Exception as e:
            logger.error(f"Failed to find callers for '{symbol_name}': {e}")
            return []
    
    def find_call_targets(self, symbol_name: str, max_depth: int = 3) -> List[RelationshipInfo]:
        """
        Find all symbols that the target symbol calls.
        
        Args:
            symbol_name: Symbol to find call targets for
            max_depth: Maximum call graph traversal depth
            
        Returns:
            List of call target relationships with full context
        """
        if not self.kg_available:
            return []
        
        try:
            symbols = self.find_symbol(symbol_name, exact_match=True)
            if not symbols:
                return []
            
            all_relationships = []
            
            for symbol in symbols:
                # Get direct call edges
                call_edges = self.db_manager.get_outgoing_edges(symbol.id, 'calls')
                
                for edge in call_edges:
                    target_node = self.db_manager.get_node(edge['target_id'])
                    if target_node:
                        target_symbol = self._node_to_symbol_info(target_node)
                        relationship = RelationshipInfo(
                            source=symbol,
                            target=target_symbol,
                            relationship_type='calls',
                            direction='outgoing',
                            depth=1,
                            metadata={'edge_data': edge}
                        )
                        all_relationships.append(relationship)
                
                # For deeper traversal, could implement recursive logic here
                # Currently keeping it simple with direct calls
            
            return all_relationships
            
        except Exception as e:
            logger.error(f"Failed to find call targets for '{symbol_name}': {e}")
            return []
    
    def trace_call_path(self, from_symbol: str, to_symbol: str, 
                       max_depth: int = 5) -> List[List[RelationshipInfo]]:
        """
        Find call paths between two symbols.
        
        Args:
            from_symbol: Starting symbol
            to_symbol: Target symbol
            max_depth: Maximum path length
            
        Returns:
            List of call paths (each path is a list of relationships)
        """
        if not self.kg_available:
            return []
        
        try:
            source_symbols = self.find_symbol(from_symbol, exact_match=True)
            target_symbols = self.find_symbol(to_symbol, exact_match=True)
            
            if not source_symbols or not target_symbols:
                return []
            
            all_paths = []
            
            for source in source_symbols:
                for target in target_symbols:
                    paths = self._find_paths_between_symbols(source, target, max_depth)
                    all_paths.extend(paths)
            
            return all_paths
            
        except Exception as e:
            logger.error(f"Failed to trace call path from '{from_symbol}' to '{to_symbol}': {e}")
            return []
    
    # ==================== DEPENDENCY ANALYSIS METHODS ====================
    
    def analyze_dependencies(self, symbol_name: str, 
                           max_depth: int = 3) -> Optional[DependencyAnalysis]:
        """
        Comprehensive dependency analysis for a symbol.
        
        Args:
            symbol_name: Symbol to analyze
            max_depth: Maximum dependency traversal depth
            
        Returns:
            Complete dependency analysis with metrics and patterns
        """
        if not self.kg_available:
            return None
        
        try:
            symbols = self.find_symbol(symbol_name, exact_match=True)
            if not symbols:
                return None
            
            # Use the first matching symbol for analysis
            symbol = symbols[0]
            
            # Get dependency data
            dependencies = self.db_manager.find_dependencies(symbol.id, max_depth)
            
            # Process dependencies into structured relationships
            direct_deps = []
            transitive_deps = []
            dependency_groups = {'calls': [], 'imports': [], 'inherits': [], 'uses': []}
            
            for dep_data in dependencies:
                dep_symbol = self._node_to_symbol_info(dep_data)
                relationship = RelationshipInfo(
                    source=symbol,
                    target=dep_symbol,
                    relationship_type=dep_data.get('relationship', 'dependency'),
                    direction='outgoing',
                    depth=dep_data.get('depth', 0),
                    metadata={'dependency_data': dep_data}
                )
                
                if dep_data.get('depth', 0) == 1:
                    direct_deps.append(relationship)
                else:
                    transitive_deps.append(relationship)
                
                # Group by relationship type
                rel_type = relationship.relationship_type
                if rel_type in dependency_groups:
                    dependency_groups[rel_type].append(relationship)
            
            # Find symbols that depend on this symbol
            dependent_symbols = []
            callers = self.db_manager.find_callers(symbol.id, max_depth)
            
            for caller_data in callers:
                caller_symbol = self._node_to_symbol_info(caller_data)
                relationship = RelationshipInfo(
                    source=caller_symbol,
                    target=symbol,
                    relationship_type='calls',
                    direction='incoming',
                    depth=caller_data.get('depth', 0),
                    metadata={'caller_data': caller_data}
                )
                dependent_symbols.append(relationship)
            
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(symbol, max_depth)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(
                direct_deps, transitive_deps, dependent_symbols
            )
            
            return DependencyAnalysis(
                target_symbol=symbol,
                direct_dependencies=direct_deps,
                transitive_dependencies=transitive_deps,
                dependent_symbols=dependent_symbols,
                dependency_groups=dependency_groups,
                circular_dependencies=circular_deps,
                complexity_metrics=complexity_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies for '{symbol_name}': {e}")
            return None
    
    def find_dependency_hotspots(self, file_pattern: str = None, 
                               min_connections: int = 5) -> List[Dict[str, Any]]:
        """
        Find symbols with high dependency connections (architectural hotspots).
        
        Args:
            file_pattern: Optional file pattern to filter symbols
            min_connections: Minimum connection count to be considered a hotspot
            
        Returns:
            List of hotspot symbols with connection metrics
        """
        if not self.kg_available:
            return []
        
        try:
            # Get all symbols (could be filtered by file pattern in future)
            hotspots = []
            
            # For now, analyze top-level symbols in each file
            # This could be optimized with a dedicated database query
            
            # Placeholder implementation - in production, would use optimized query
            logger.info("Dependency hotspot analysis requires additional optimization")
            return []
            
        except Exception as e:
            logger.error(f"Failed to find dependency hotspots: {e}")
            return []
    
    # ==================== ARCHITECTURAL ANALYSIS METHODS ====================
    
    def analyze_module_coupling(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze coupling between modules/files.
        
        Args:
            module_path: Path to module to analyze
            
        Returns:
            Module coupling analysis with metrics
        """
        if not self.kg_available:
            return {}
        
        try:
            # Get all symbols in the module
            module_symbols = self.db_manager.get_nodes_by_file(module_path)
            
            if not module_symbols:
                return {'error': f"No symbols found in module '{module_path}'"}
            
            # Analyze external dependencies
            external_dependencies = set()
            internal_relationships = 0
            
            for symbol_data in module_symbols:
                symbol_id = symbol_data['id']
                
                # Check outgoing relationships
                all_edges = []
                all_edges.extend(self.db_manager.get_outgoing_edges(symbol_id, 'calls'))
                all_edges.extend(self.db_manager.get_outgoing_edges(symbol_id, 'imports'))
                all_edges.extend(self.db_manager.get_outgoing_edges(symbol_id, 'inherits'))
                
                for edge in all_edges:
                    target_node = self.db_manager.get_node(edge['target_id'])
                    if target_node:
                        if target_node['file_path'] != module_path:
                            external_dependencies.add(target_node['file_path'])
                        else:
                            internal_relationships += 1
            
            # Calculate coupling metrics
            coupling_metrics = {
                'total_symbols': len(module_symbols),
                'external_dependencies': len(external_dependencies),
                'internal_relationships': internal_relationships,
                'external_dependency_files': list(external_dependencies),
                'coupling_ratio': len(external_dependencies) / max(len(module_symbols), 1),
                'cohesion_ratio': internal_relationships / max(len(module_symbols), 1)
            }
            
            return {
                'module_path': module_path,
                'coupling_analysis': coupling_metrics,
                'recommendations': self._generate_coupling_recommendations(coupling_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze module coupling for '{module_path}': {e}")
            return {'error': str(e)}
    
    def find_inheritance_chains(self, base_class: str = None) -> List[List[SymbolInfo]]:
        """
        Find inheritance chains in the codebase.
        
        Args:
            base_class: Optional base class to start from
            
        Returns:
            List of inheritance chains (each chain is a list of symbols)
        """
        if not self.kg_available:
            return []
        
        try:
            inheritance_chains = []
            
            if base_class:
                # Find specific inheritance chains from base class
                base_symbols = self.find_symbol(base_class, exact_match=True)
                for symbol in base_symbols:
                    chains = self._trace_inheritance_chains(symbol)
                    inheritance_chains.extend(chains)
            else:
                # Find all inheritance chains (could be expensive)
                logger.info("Full inheritance chain analysis not yet optimized for large codebases")
                return []
            
            return inheritance_chains
            
        except Exception as e:
            logger.error(f"Failed to find inheritance chains: {e}")
            return []
    
    # ==================== UTILITY AND CONVERSION METHODS ====================
    
    def _node_to_symbol_info(self, node_data: Dict) -> SymbolInfo:
        """Convert database node data to SymbolInfo object."""
        return SymbolInfo(
            id=node_data['id'],
            name=node_data['name'],
            type=node_data['type'],
            file_path=node_data['file_path'],
            line_start=node_data.get('line_start', 0),
            line_end=node_data.get('line_end', 0),
            signature=node_data.get('signature'),
            docstring=node_data.get('docstring'),
            metadata=node_data.get('metadata', {})
        )
    
    def _analyze_symbol_neighborhood(self, symbol: SymbolInfo, 
                                   max_depth: int, include_siblings: bool) -> Dict[str, Any]:
        """Analyze the neighborhood of relationships around a symbol."""
        neighborhood = {
            'center_symbol': symbol.to_dict(),
            'callers': [],
            'call_targets': [],
            'dependencies': [],
            'inheritance': {'parents': [], 'children': []},
            'imports': {'imported_by': [], 'imports_from': []},
            'file_siblings': [],
            'metrics': {}
        }
        
        try:
            # Find callers
            callers = self.db_manager.find_callers(symbol.id, max_depth)
            neighborhood['callers'] = [self._node_to_symbol_info(c).to_dict() for c in callers[:10]]
            
            # Find call targets
            call_edges = self.db_manager.get_outgoing_edges(symbol.id, 'calls')
            for edge in call_edges[:10]:
                target_node = self.db_manager.get_node(edge['target_id'])
                if target_node:
                    neighborhood['call_targets'].append(self._node_to_symbol_info(target_node).to_dict())
            
            # Find dependencies
            dependencies = self.db_manager.find_dependencies(symbol.id, max_depth)
            neighborhood['dependencies'] = [self._node_to_symbol_info(d).to_dict() for d in dependencies[:10]]
            
            # Find inheritance relationships
            inherit_edges = self.db_manager.get_outgoing_edges(symbol.id, 'inherits')
            for edge in inherit_edges:
                parent_node = self.db_manager.get_node(edge['target_id'])
                if parent_node:
                    neighborhood['inheritance']['parents'].append(self._node_to_symbol_info(parent_node).to_dict())
            
            inherited_by_edges = self.db_manager.get_incoming_edges(symbol.id, 'inherits')
            for edge in inherited_by_edges:
                child_node = self.db_manager.get_node(edge['source_id'])
                if child_node:
                    neighborhood['inheritance']['children'].append(self._node_to_symbol_info(child_node).to_dict())
            
            # Find import relationships
            import_edges = self.db_manager.get_outgoing_edges(symbol.id, 'imports')
            for edge in import_edges:
                imported_node = self.db_manager.get_node(edge['target_id'])
                if imported_node:
                    neighborhood['imports']['imports_from'].append(self._node_to_symbol_info(imported_node).to_dict())
            
            imported_by_edges = self.db_manager.get_incoming_edges(symbol.id, 'imports')
            for edge in imported_by_edges:
                importer_node = self.db_manager.get_node(edge['source_id'])
                if importer_node:
                    neighborhood['imports']['imported_by'].append(self._node_to_symbol_info(importer_node).to_dict())
            
            # Find file siblings
            if include_siblings:
                file_symbols = self.db_manager.get_nodes_by_file(symbol.file_path)
                siblings = [s for s in file_symbols if s['id'] != symbol.id]
                neighborhood['file_siblings'] = [self._node_to_symbol_info(s).to_dict() for s in siblings[:8]]
            
            # Calculate metrics
            neighborhood['metrics'] = {
                'total_callers': len(callers),
                'total_call_targets': len(call_edges),
                'total_dependencies': len(dependencies),
                'inheritance_depth': len(neighborhood['inheritance']['parents']),
                'inheritance_breadth': len(neighborhood['inheritance']['children']),
                'coupling_factor': len(callers) + len(call_edges) + len(dependencies),
                'file_siblings_count': len(neighborhood['file_siblings'])
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze neighborhood for symbol {symbol.name}: {e}")
            neighborhood['error'] = str(e)
        
        return neighborhood
    
    def _generate_neighborhood_summary(self, neighborhoods: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary statistics across all analyzed neighborhoods."""
        total_relationships = 0
        total_symbols = len(neighborhoods)
        relationship_types = set()
        
        for neighborhood in neighborhoods.values():
            if 'metrics' in neighborhood:
                metrics = neighborhood['metrics']
                total_relationships += metrics.get('coupling_factor', 0)
            
            # Count relationship types
            for key in ['callers', 'call_targets', 'dependencies']:
                if neighborhood.get(key):
                    relationship_types.add(key)
        
        return {
            'total_symbols_analyzed': total_symbols,
            'total_relationships_found': total_relationships,
            'avg_relationships_per_symbol': total_relationships / max(total_symbols, 1),
            'relationship_types_found': list(relationship_types)
        }
    
    def _find_paths_between_symbols(self, source: SymbolInfo, target: SymbolInfo, 
                                   max_depth: int) -> List[List[RelationshipInfo]]:
        """Find call paths between two symbols using graph traversal."""
        # This would implement a sophisticated path-finding algorithm
        # For now, returning empty list - could be implemented with BFS/DFS
        logger.debug(f"Path finding from {source.name} to {target.name} not yet implemented")
        return []
    
    def _detect_circular_dependencies(self, symbol: SymbolInfo, max_depth: int) -> List[List[str]]:
        """Detect circular dependencies involving the symbol."""
        # This would implement cycle detection in the dependency graph
        # For now, returning empty list - could be implemented with DFS cycle detection
        logger.debug(f"Circular dependency detection for {symbol.name} not yet implemented")
        return []
    
    def _calculate_complexity_metrics(self, direct_deps: List[RelationshipInfo],
                                    transitive_deps: List[RelationshipInfo],
                                    dependents: List[RelationshipInfo]) -> Dict[str, Any]:
        """Calculate complexity metrics for dependency analysis."""
        return {
            'direct_dependency_count': len(direct_deps),
            'transitive_dependency_count': len(transitive_deps),
            'dependent_count': len(dependents),
            'total_complexity': len(direct_deps) + len(transitive_deps) + len(dependents),
            'fan_in': len(dependents),  # How many depend on this
            'fan_out': len(direct_deps),  # How many this depends on
            'instability': len(direct_deps) / max(len(direct_deps) + len(dependents), 1)
        }
    
    def _generate_coupling_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coupling analysis."""
        recommendations = []
        
        coupling_ratio = metrics.get('coupling_ratio', 0)
        cohesion_ratio = metrics.get('cohesion_ratio', 0)
        
        if coupling_ratio > 0.8:
            recommendations.append("High external coupling detected - consider reducing dependencies")
        
        if cohesion_ratio < 0.3:
            recommendations.append("Low internal cohesion - consider splitting module")
        
        if len(metrics.get('external_dependency_files', [])) > 10:
            recommendations.append("Many external dependencies - consider dependency injection or facades")
        
        if not recommendations:
            recommendations.append("Module coupling appears healthy")
        
        return recommendations
    
    def _trace_inheritance_chains(self, base_symbol: SymbolInfo) -> List[List[SymbolInfo]]:
        """Trace inheritance chains from a base symbol."""
        # This would implement inheritance chain traversal
        # For now, returning empty list - could be implemented with recursive traversal
        logger.debug(f"Inheritance chain tracing for {base_symbol.name} not yet implemented")
        return []
    
    def close(self):
        """Clean up resources."""
        if self.db_manager:
            self.db_manager.close()


# ==================== TOOL-FRIENDLY QUERY FUNCTIONS ====================

def kg_find_symbol(symbol_name: str, exact_match: bool = True, 
                  db_path: str = "codewise.db") -> str:
    """
    Tool-friendly function to find symbols in Knowledge Graph.
    
    Args:
        symbol_name: Name of symbol to find
        exact_match: Whether to use exact matching
        db_path: Path to KG database
        
    Returns:
        JSON string with symbol information
    """
    query_methods = KGQueryMethods(db_path)
    
    try:
        symbols = query_methods.find_symbol(symbol_name, exact_match)
        
        if not symbols:
            return json.dumps({
                'error': f"Symbol '{symbol_name}' not found in Knowledge Graph",
                'suggestion': "Try using exact_match=False for fuzzy search"
            })
        
        result = {
            'symbols_found': len(symbols),
            'symbols': [symbol.to_dict() for symbol in symbols[:10]]  # Limit to top 10
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        query_methods.close()


def kg_explore_neighborhood(symbol_name: str, max_depth: int = 3,
                          db_path: str = "codewise.db") -> str:
    """
    Tool-friendly function to explore symbol neighborhood.
    
    Args:
        symbol_name: Symbol to explore
        max_depth: Maximum relationship depth
        db_path: Path to KG database
        
    Returns:
        JSON string with neighborhood analysis
    """
    query_methods = KGQueryMethods(db_path)
    
    try:
        neighborhood = query_methods.explore_symbol_neighborhood(symbol_name, max_depth)
        return json.dumps(neighborhood, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        query_methods.close()


def kg_analyze_dependencies(symbol_name: str, max_depth: int = 3,
                          db_path: str = "codewise.db") -> str:
    """
    Tool-friendly function for dependency analysis.
    
    Args:
        symbol_name: Symbol to analyze
        max_depth: Maximum dependency depth
        db_path: Path to KG database
        
    Returns:
        JSON string with dependency analysis
    """
    query_methods = KGQueryMethods(db_path)
    
    try:
        analysis = query_methods.analyze_dependencies(symbol_name, max_depth)
        
        if not analysis:
            return json.dumps({
                'error': f"Could not analyze dependencies for '{symbol_name}'",
                'suggestion': "Check if symbol exists in Knowledge Graph"
            })
        
        return json.dumps(analysis.to_dict(), indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        query_methods.close()


def kg_find_callers(symbol_name: str, max_depth: int = 3,
                   db_path: str = "codewise.db") -> str:
    """
    Tool-friendly function to find symbol callers.
    
    Args:
        symbol_name: Symbol to find callers for
        max_depth: Maximum call graph depth
        db_path: Path to KG database
        
    Returns:
        JSON string with caller information
    """
    query_methods = KGQueryMethods(db_path)
    
    try:
        callers = query_methods.find_callers(symbol_name, max_depth)
        
        result = {
            'symbol_name': symbol_name,
            'callers_found': len(callers),
            'callers': [caller.to_dict() for caller in callers[:20]]  # Limit to top 20
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        query_methods.close()


if __name__ == "__main__":
    # CLI interface for testing KG Query Methods
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KG Query Methods")
    parser.add_argument("--db-path", default="codewise.db", help="Database file path")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Find symbol command
    find_parser = subparsers.add_parser('find', help='Find symbol')
    find_parser.add_argument('symbol', help='Symbol name to find')
    find_parser.add_argument('--fuzzy', action='store_true', help='Use fuzzy matching')
    
    # Explore neighborhood command
    explore_parser = subparsers.add_parser('explore', help='Explore symbol neighborhood')
    explore_parser.add_argument('symbol', help='Symbol to explore')
    explore_parser.add_argument('--depth', type=int, default=3, help='Max relationship depth')
    
    # Analyze dependencies command
    deps_parser = subparsers.add_parser('deps', help='Analyze dependencies')
    deps_parser.add_argument('symbol', help='Symbol to analyze')
    deps_parser.add_argument('--depth', type=int, default=3, help='Max dependency depth')
    
    # Find callers command
    callers_parser = subparsers.add_parser('callers', help='Find callers')
    callers_parser.add_argument('symbol', help='Symbol to find callers for')
    callers_parser.add_argument('--depth', type=int, default=3, help='Max call graph depth')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Execute command
    if args.command == 'find':
        result = kg_find_symbol(args.symbol, not args.fuzzy, args.db_path)
        print(result)
    
    elif args.command == 'explore':
        result = kg_explore_neighborhood(args.symbol, args.depth, args.db_path)
        print(result)
    
    elif args.command == 'deps':
        result = kg_analyze_dependencies(args.symbol, args.depth, args.db_path)
        print(result)
    
    elif args.command == 'callers':
        result = kg_find_callers(args.symbol, args.depth, args.db_path)
        print(result)