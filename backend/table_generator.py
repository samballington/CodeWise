#!/usr/bin/env python3
"""
Table Generator Utility for CodeWise
Provides standardized JSON table generation using the codewise_structured_v1 format
"""

from typing import List, Dict, Any, Optional, Union
import json
from dataclasses import dataclass, asdict
from enum import Enum

@dataclass
class StructuredTable:
    """Represents a structured table for UI rendering"""
    title: Optional[str] = None
    columns: List[str] = None
    rows: List[List[Union[str, int, float, None]]] = None  
    note: Optional[str] = None
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.rows is None:
            self.rows = []

@dataclass 
class StructuredTree:
    """Represents a structured tree for UI rendering"""
    title: Optional[str] = None
    root: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.root is None:
            self.root = {"label": "", "children": []}

@dataclass
class FileReference:
    """File reference with optional line numbers"""
    path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None

class TableGenerator:
    """
    Utility for generating standardized JSON tables in codewise_structured_v1 format
    """
    
    @staticmethod
    def create_search_results_table(results: List[Dict[str, Any]], 
                                  title: str = "Search Results") -> StructuredTable:
        """
        Convert search results to structured table format
        
        Args:
            results: List of search result dictionaries
            title: Table title
            
        Returns:
            StructuredTable object
        """
        if not results:
            return StructuredTable(
                title=title,
                columns=["Status"],
                rows=[["No results found"]],
                note="Try refining your search query"
            )
        
        columns = ["File", "Relevance", "Snippet"]
        rows = []
        
        for result in results[:10]:  # Limit to 10 results for table
            file_path = result.get('file_path', 'Unknown')
            relevance = result.get('relevance_score', 0.0)
            snippet = result.get('snippet', '').strip()[:100]  # Truncate for table
            
            # Format relevance as percentage
            relevance_str = f"{relevance:.1%}" if isinstance(relevance, (int, float)) else str(relevance)
            
            # Clean up snippet for table display
            if snippet:
                snippet = snippet.replace('\n', ' ').replace('\t', ' ')
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
            else:
                snippet = "(no snippet)"
                
            rows.append([file_path, relevance_str, snippet])
        
        note = f"Found {len(results)} total results" if len(results) > 10 else None
        
        return StructuredTable(
            title=title,
            columns=columns,
            rows=rows,
            note=note
        )
    
    @staticmethod
    def create_file_analysis_table(files: List[Dict[str, Any]], 
                                 title: str = "File Analysis") -> StructuredTable:
        """
        Convert file analysis results to structured table format
        
        Args:
            files: List of file analysis dictionaries
            title: Table title
            
        Returns:
            StructuredTable object
        """
        if not files:
            return StructuredTable(
                title=title,
                columns=["Status"],
                rows=[["No files analyzed"]],
                note="No files match the analysis criteria"
            )
        
        columns = ["File", "Type", "Size", "Last Modified"]
        rows = []
        
        for file_info in files[:15]:  # Limit to 15 files for table
            file_path = file_info.get('path', 'Unknown')
            file_type = file_info.get('type', 'Unknown')
            file_size = file_info.get('size', 'Unknown')
            last_modified = file_info.get('last_modified', 'Unknown')
            
            # Format file size
            if isinstance(file_size, (int, float)):
                if file_size > 1024 * 1024:
                    file_size = f"{file_size / (1024*1024):.1f} MB"
                elif file_size > 1024:
                    file_size = f"{file_size / 1024:.1f} KB"
                else:
                    file_size = f"{file_size} B"
            
            rows.append([file_path, file_type, file_size, last_modified])
        
        note = f"Analyzed {len(files)} total files" if len(files) > 15 else None
        
        return StructuredTable(
            title=title,
            columns=columns,
            rows=rows,
            note=note
        )
    
    @staticmethod
    def create_dependency_table(dependencies: List[Dict[str, Any]], 
                               title: str = "Dependencies") -> StructuredTable:
        """
        Convert dependency analysis to structured table format
        
        Args:
            dependencies: List of dependency dictionaries
            title: Table title
            
        Returns:
            StructuredTable object
        """
        if not dependencies:
            return StructuredTable(
                title=title,
                columns=["Status"],
                rows=[["No dependencies found"]],
                note="No dependency relationships detected"
            )
        
        columns = ["Package", "Version", "Type", "Status"]
        rows = []
        
        for dep in dependencies[:20]:  # Limit to 20 dependencies
            package = dep.get('name', 'Unknown')
            version = dep.get('version', 'Unknown')
            dep_type = dep.get('type', 'Unknown')
            status = dep.get('status', 'Unknown')
            
            rows.append([package, version, dep_type, status])
        
        note = f"Found {len(dependencies)} total dependencies" if len(dependencies) > 20 else None
        
        return StructuredTable(
            title=title,
            columns=columns,
            rows=rows,
            note=note
        )
    
    @staticmethod
    def create_project_tree(structure: Dict[str, Any], 
                           title: str = "Project Structure") -> StructuredTree:
        """
        Convert project structure to structured tree format
        
        Args:
            structure: Project structure dictionary
            title: Tree title
            
        Returns:
            StructuredTree object
        """
        def convert_to_tree_node(node_data: Dict[str, Any]) -> Dict[str, Any]:
            """Convert a node to tree format"""
            if isinstance(node_data, dict):
                label = node_data.get('name', node_data.get('label', 'Unknown'))
                children = []
                
                # Handle children if present
                if 'children' in node_data:
                    for child in node_data['children']:
                        children.append(convert_to_tree_node(child))
                elif 'files' in node_data:
                    for file_item in node_data['files']:
                        children.append(convert_to_tree_node(file_item))
                
                return {
                    "label": str(label),
                    "children": children
                }
            else:
                return {
                    "label": str(node_data),
                    "children": []
                }
        
        # Convert structure to tree format
        if structure:
            root = convert_to_tree_node(structure)
        else:
            root = {
                "label": "No structure available",
                "children": []
            }
        
        return StructuredTree(
            title=title,
            root=root
        )
    
    @staticmethod
    def create_comparison_table(items: List[Dict[str, Any]], 
                              compare_fields: List[str],
                              title: str = "Comparison") -> StructuredTable:
        """
        Create a comparison table for similar items
        
        Args:
            items: List of items to compare
            compare_fields: Fields to include in comparison
            title: Table title
            
        Returns:
            StructuredTable object
        """
        if not items or not compare_fields:
            return StructuredTable(
                title=title,
                columns=["Status"],
                rows=[["No items to compare"]],
                note="Provide items and comparison fields"
            )
        
        # Use compare_fields as columns
        columns = compare_fields
        rows = []
        
        for item in items[:10]:  # Limit to 10 items
            row = []
            for field in compare_fields:
                value = item.get(field, 'N/A')
                # Convert to string and truncate if needed
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                row.append(value_str)
            rows.append(row)
        
        note = f"Comparing {len(items)} items across {len(compare_fields)} fields"
        
        return StructuredTable(
            title=title,
            columns=columns,
            rows=rows,
            note=note
        )
    
    @staticmethod
    def generate_structured_response(tables: List[StructuredTable] = None,
                                   trees: List[StructuredTree] = None,
                                   references: List[FileReference] = None) -> str:
        """
        Generate the complete JSON structured response
        
        Args:
            tables: List of StructuredTable objects
            trees: List of StructuredTree objects  
            references: List of FileReference objects
            
        Returns:
            JSON string in codewise_structured_v1 format
        """
        if tables is None:
            tables = []
        if trees is None:
            trees = []
        if references is None:
            references = []
        
        # Convert to dictionaries
        tables_data = [asdict(table) for table in tables]
        trees_data = [asdict(tree) for tree in trees]
        references_data = [asdict(ref) for ref in references]
        
        structured_response = {
            "version": "codewise_structured_v1",
            "tables": tables_data,
            "trees": trees_data,
            "references": references_data
        }
        
        return json.dumps(structured_response, indent=2)
    
    @staticmethod
    def wrap_response_with_structured_data(markdown_response: str,
                                         tables: List[StructuredTable] = None,
                                         trees: List[StructuredTree] = None,
                                         references: List[FileReference] = None,
                                         enable_deduplication: bool = True,
                                         deduplication_timeout: float = 0.1) -> str:
        """
        Wrap a markdown response with structured JSON data, optionally removing duplicate ASCII tables
        
        Args:
            markdown_response: The main markdown response
            tables: List of tables to include
            trees: List of trees to include
            references: List of file references
            enable_deduplication: Whether to remove duplicate ASCII tables from markdown
            deduplication_timeout: Maximum time to spend on deduplication (seconds)
            
        Returns:
            Complete response with markdown + JSON block (and duplicate tables removed if enabled)
        """
        if not tables and not trees and not references:
            return markdown_response
        
        # Apply deduplication if enabled and we have tables
        cleaned_markdown = markdown_response
        if enable_deduplication and tables:
            try:
                import time
                start_time = time.time()
                
                # Import here to avoid circular imports
                from table_matcher import TableMatcher
                from table_remover import TableRemover
                
                # Convert StructuredTable objects to dict format for matcher
                table_dicts = []
                for table in tables:
                    table_dict = {
                        'title': table.title,
                        'columns': table.columns or [],
                        'rows': table.rows or []
                    }
                    table_dicts.append(table_dict)
                
                # Find and remove duplicate tables with timeout protection
                matcher = TableMatcher(threshold=0.7)
                matches = matcher.find_duplicate_tables(markdown_response, table_dicts)
                
                # Check timeout
                if time.time() - start_time > deduplication_timeout:
                    import logging
                    logging.warning(f"Table deduplication timed out after {deduplication_timeout}s")
                else:
                    if matches:
                        remover = TableRemover()
                        cleaned_markdown = remover.remove_duplicate_tables(markdown_response, matches)
                        
                        import logging
                        logging.info(f"Removed {len(matches)} duplicate ASCII tables")
                
            except Exception as e:
                # Graceful degradation - if deduplication fails, continue with original markdown
                import logging
                logging.warning(f"Table deduplication failed: {e}")
                cleaned_markdown = markdown_response
        
        json_block = TableGenerator.generate_structured_response(tables, trees, references)
        
        return f"{cleaned_markdown}\n\n```json\n{json_block}\n```"


class TablePresets:
    """Common table presets for frequently used table types"""
    
    @staticmethod
    def empty_results_table(message: str = "No results found") -> StructuredTable:
        """Standard empty results table"""
        return StructuredTable(
            columns=["Status"],
            rows=[[message]],
            note="Try refining your search query"
        )
    
    @staticmethod
    def error_table(error_message: str) -> StructuredTable:
        """Standard error table"""
        return StructuredTable(
            title="Error",
            columns=["Error"],
            rows=[[error_message]],
            note="Please try again or contact support"
        )
    
    @staticmethod 
    def loading_table() -> StructuredTable:
        """Standard loading table"""
        return StructuredTable(
            columns=["Status"],
            rows=[["Loading..."]],
            note="Please wait while data is being processed"
        )