"""
Universal Relationship Engine Integration Layer

Integrates the Universal Pattern Recognizer with existing KG and vector search systems
to provide comprehensive relationship detection for diagram generation.

This module serves as the main entry point for relationship inference across all 
programming languages and frameworks.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .universal_pattern_recognizer import (
    UniversalPatternRecognizer, 
    UniversalRelationshipSynthesizer,
    DetectedRelationship,
    RelationshipType
)

logger = logging.getLogger(__name__)

class UniversalRelationshipEngine:
    """
    Main integration engine for universal relationship detection.
    
    Coordinates between:
    - Universal Pattern Recognizer (source code analysis)
    - Knowledge Graph (structural relationships)
    - Vector Search (contextual relationships)
    - File System Analysis (architectural patterns)
    """
    
    def __init__(self):
        self.pattern_recognizer = UniversalPatternRecognizer()
        self.relationship_synthesizer = UniversalRelationshipSynthesizer()
        
    async def infer_relationships_from_components(self, components: List[Dict],
                                                query_context: str = "") -> List[DetectedRelationship]:
        """
        Main entry point for inferring relationships from discovered components.
        
        Args:
            components: List of components from KG/vector search
            query_context: Original query for context
            
        Returns:
            List of detected relationships with confidence scores
        """
        try:
            logger.info(f"🔗 Starting universal relationship inference for {len(components)} components")
            
            # Extract source files from components
            source_files = self._extract_source_files(components)
            
            # Perform multi-source relationship detection
            kg_relationships = await self._extract_kg_relationships(components)
            vector_relationships = await self._extract_vector_relationships(components, query_context)
            code_relationships = await self._extract_code_relationships(source_files)
            
            # Synthesize all relationships with confidence scoring
            synthesized_relationships = self._synthesize_all_relationships(
                kg_relationships, vector_relationships, code_relationships, components)
            
            # Filter and rank by relevance to query
            relevant_relationships = self._filter_relevant_relationships(
                synthesized_relationships, query_context)
            
            logger.info(f"✅ Inferred {len(relevant_relationships)} high-confidence relationships")
            return relevant_relationships
            
        except Exception as e:
            logger.error(f"❌ Universal relationship inference failed: {e}")
            return []
    
    def _extract_source_files(self, components: List[Dict]) -> List[Dict]:
        """Extract source file information from components"""
        source_files = []
        
        for component in components:
            # Handle different component formats
            if 'node_data' in component:
                # KG structural node format
                node = component['node_data']
                if node.get('file_path'):
                    source_files.append({
                        'file_path': node['file_path'],
                        'component_name': node.get('name', 'unknown'),
                        'component_type': node.get('type', 'unknown')
                    })
            elif 'file_path' in component:
                # Direct file path format
                source_files.append({
                    'file_path': component['file_path'],
                    'component_name': component.get('name', Path(component['file_path']).stem),
                    'component_type': component.get('type', 'file')
                })
            elif 'data' in component and component['data'].get('file_path'):
                # Nested data format
                data = component['data']
                source_files.append({
                    'file_path': data['file_path'],
                    'component_name': data.get('name', 'unknown'),
                    'component_type': data.get('type', 'unknown')
                })
        
        # Deduplicate by file path
        seen_paths = set()
        unique_files = []
        for file_info in source_files:
            if file_info['file_path'] not in seen_paths:
                seen_paths.add(file_info['file_path'])
                unique_files.append(file_info)
        
        logger.info(f"📁 Extracted {len(unique_files)} unique source files for analysis")
        return unique_files
    
    async def _extract_kg_relationships(self, components: List[Dict]) -> List[Dict]:
        """Extract existing relationships from Knowledge Graph"""
        kg_relationships = []
        
        try:
            from storage.database_manager import DatabaseManager
            db = DatabaseManager()
            cursor = db.connection.cursor()
            
            # Collect all component IDs
            component_ids = set()
            for component in components:
                if 'node_data' in component:
                    component_ids.add(component['node_data'].get('id'))
                elif 'data' in component:
                    component_ids.add(component['data'].get('id'))
                elif 'id' in component:
                    component_ids.add(component['id'])
            
            # Remove None values
            component_ids = [id for id in component_ids if id is not None]
            
            if component_ids:
                # Query for relationships between these components
                placeholders = ','.join(['?' for _ in component_ids])
                query = f"""
                    SELECT source_id, target_id, type, properties
                    FROM edges 
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                """
                params = component_ids + component_ids
                
                rows = cursor.execute(query, params).fetchall()
                
                for row in rows:
                    source_id, target_id, rel_type, properties = row
                    kg_relationships.append({
                        'source_id': source_id,
                        'target_id': target_id,
                        'type': rel_type,
                        'properties': properties
                    })
            
            logger.info(f"📊 Extracted {len(kg_relationships)} KG relationships")
            
        except Exception as e:
            logger.warning(f"KG relationship extraction failed: {e}")
        
        return kg_relationships
    
    async def _extract_vector_relationships(self, components: List[Dict], 
                                          query_context: str) -> List[Dict]:
        """Extract relationships using vector search context"""
        vector_relationships = []
        
        try:
            # Use vector search to find contextual relationships
            from vector_store import get_vector_store
            vs = get_vector_store()
            
            # Search for relationship context between components
            component_names = []
            for component in components:
                if 'node_data' in component:
                    component_names.append(component['node_data'].get('name', ''))
                elif 'data' in component:
                    component_names.append(component['data'].get('name', ''))
                elif 'name' in component:
                    component_names.append(component['name'])
            
            # Create relationship search queries
            relationship_queries = []
            for i, source in enumerate(component_names):
                for target in component_names[i+1:]:
                    if source and target:
                        relationship_queries.append(f"{source} {target} dependency relationship")
            
            # Execute vector searches for relationship context
            for query in relationship_queries[:10]:  # Limit to prevent explosion
                try:
                    results = await vs.similarity_search(query, k=3)
                    for result in results:
                        if result.get('score', 0) > 0.7:  # High similarity threshold
                            vector_relationships.append({
                                'source': query.split()[0],
                                'target': query.split()[1],
                                'context': result.get('content', ''),
                                'score': result.get('score', 0.0)
                            })
                except Exception as e:
                    logger.debug(f"Vector search failed for query '{query}': {e}")
            
            logger.info(f"🔍 Extracted {len(vector_relationships)} vector relationships")
            
        except Exception as e:
            logger.warning(f"Vector relationship extraction failed: {e}")
        
        return vector_relationships
    
    async def _extract_code_relationships(self, source_files: List[Dict]) -> List[DetectedRelationship]:
        """Extract relationships through source code analysis"""
        code_relationships = []
        
        for file_info in source_files:
            try:
                file_path = file_info['file_path']
                
                # Read source file
                if not Path(file_path).exists():
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()
                
                # Detect language from file extension
                language = self._detect_language(file_path)
                
                # Run pattern recognition
                relationships = self.pattern_recognizer.recognize_all_patterns(
                    source_code, language, file_path)
                
                code_relationships.extend(relationships)
                
            except Exception as e:
                logger.warning(f"Code analysis failed for {file_info['file_path']}: {e}")
        
        logger.info(f"💻 Extracted {len(code_relationships)} code relationships")
        return code_relationships
    
    def _synthesize_all_relationships(self, kg_relationships: List[Dict],
                                    vector_relationships: List[Dict],
                                    code_relationships: List[DetectedRelationship],
                                    components: List[Dict]) -> List[DetectedRelationship]:
        """Synthesize relationships from all sources"""
        
        # Detect primary language from components
        language = self._detect_primary_language(components)
        
        # Use relationship synthesizer
        synthesized = self.relationship_synthesizer.synthesize_relationships(
            kg_relationships, vector_relationships, code_relationships, language)
        
        return synthesized
    
    def _filter_relevant_relationships(self, relationships: List[DetectedRelationship],
                                     query_context: str) -> List[DetectedRelationship]:
        """Filter relationships by relevance to query and confidence threshold"""
        
        # Filter by minimum confidence
        high_confidence = [r for r in relationships if r.confidence >= 0.6]
        
        # Sort by confidence
        high_confidence.sort(key=lambda r: r.confidence, reverse=True)
        
        # Limit to prevent diagram complexity explosion
        return high_confidence[:50]
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'javascript',
            '.jsx': 'javascript',
            '.tsx': 'javascript',
            '.go': 'go',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.kt': 'kotlin'
        }
        
        return language_map.get(ext, 'unknown')
    
    def _detect_primary_language(self, components: List[Dict]) -> str:
        """Detect primary language from component file paths"""
        language_counts = {}
        
        for component in components:
            file_path = None
            if 'node_data' in component:
                file_path = component['node_data'].get('file_path')
            elif 'file_path' in component:
                file_path = component['file_path']
            elif 'data' in component:
                file_path = component['data'].get('file_path')
            
            if file_path:
                language = self._detect_language(file_path)
                language_counts[language] = language_counts.get(language, 0) + 1
        
        # Return most common language
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    def convert_to_graph_format(self, relationships: List[DetectedRelationship]) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert detected relationships to graph format for diagram rendering.
        
        Returns:
            Tuple of (nodes, edges) for diagram generation
        """
        nodes = {}
        edges = []
        
        for rel in relationships:
            # Add source node
            if rel.source_component not in nodes:
                nodes[rel.source_component] = {
                    'id': rel.source_component,
                    'name': rel.source_component,
                    'type': self._infer_node_type(rel.source_component),
                    'language': rel.source_language
                }
            
            # Add target node
            if rel.target_component not in nodes:
                nodes[rel.target_component] = {
                    'id': rel.target_component,
                    'name': rel.target_component,
                    'type': self._infer_node_type(rel.target_component),
                    'language': rel.source_language
                }
            
            # Add edge
            edges.append({
                'source': rel.source_component,
                'target': rel.target_component,
                'type': rel.relationship_type.value,
                'confidence': rel.confidence,
                'label': self._get_relationship_label(rel.relationship_type),
                'metadata': rel.metadata or {}
            })
        
        return list(nodes.values()), edges
    
    def _infer_node_type(self, component_name: str) -> str:
        """Infer node type from component name"""
        name_lower = component_name.lower()
        
        if any(x in name_lower for x in ['controller', 'handler']):
            return 'controller'
        elif any(x in name_lower for x in ['service', 'business']):
            return 'service'
        elif any(x in name_lower for x in ['repository', 'dao', 'data']):
            return 'repository'
        elif any(x in name_lower for x in ['model', 'entity']):
            return 'model'
        elif any(x in name_lower for x in ['component', 'widget']):
            return 'component'
        else:
            return 'class'
    
    def _get_relationship_label(self, relationship_type: RelationshipType) -> str:
        """Get display label for relationship type"""
        labels = {
            RelationshipType.DEPENDENCY_INJECTION: 'uses',
            RelationshipType.INTERFACE_IMPLEMENTATION: 'implements',
            RelationshipType.INHERITANCE: 'extends',
            RelationshipType.COMPOSITION: 'contains',
            RelationshipType.AGGREGATION: 'has',
            RelationshipType.EVENT_SUBSCRIPTION: 'subscribes',
            RelationshipType.CONFIGURATION_DEPENDENCY: 'configures',
            RelationshipType.API_CALL: 'calls',
            RelationshipType.MODULE_IMPORT: 'imports',
            RelationshipType.DATABASE_ACCESS: 'accesses'
        }
        
        return labels.get(relationship_type, 'relates')


# Export the main integration class
__all__ = ["UniversalRelationshipEngine", "DetectedRelationship", "RelationshipType"]