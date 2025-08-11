#!/usr/bin/env python3
"""
Markdown Table Processor for CodeWise
Handles detection and manipulation of ASCII tables in markdown content
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AsciiTableInfo:
    """Information about an ASCII table found in markdown"""
    start_line: int
    end_line: int
    content: str
    title: Optional[str]
    columns: List[str]
    row_count: int
    raw_lines: List[str]
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.raw_lines is None:
            self.raw_lines = []

@dataclass
class TableMetadata:
    """Metadata extracted from a table for matching"""
    title: Optional[str]
    column_names: List[str]
    column_count: int
    row_count: int
    data_sample: List[str]  # First few data values for similarity matching
    
    def __post_init__(self):
        if self.column_names is None:
            self.column_names = []
        if self.data_sample is None:
            self.data_sample = []

class MarkdownTableProcessor:
    """Utility for detecting and manipulating ASCII tables in markdown content"""
    
    # Regex patterns for different ASCII table formats
    TABLE_ROW_PATTERNS = [
        # Standard GFM table row: | col1 | col2 | col3 |
        re.compile(r'^\s*\|(.+\|)+\s*$'),
        # Loose table row: |col1|col2|col3|
        re.compile(r'^\s*\|[^|]*(\|[^|]*)+\|\s*$'),
    ]
    
    # Pattern for GFM header divider: |------|------| or |------|
    DIVIDER_PATTERN = re.compile(r'^\s*\|?\s*:?[- ]+:?\s*(\|\s*:?[- ]+:?\s*)*\|?\s*$')
    
    # Pattern for potential table titles (headers before tables)
    TITLE_PATTERNS = [
        re.compile(r'^#+\s+(.+)$'),  # Markdown headers
        re.compile(r'^(.+)\s*$'),    # Any line before table (fallback)
    ]
    
    @staticmethod
    def detect_ascii_tables(markdown: str) -> List[AsciiTableInfo]:
        """
        Detect all ASCII tables in markdown content
        
        Args:
            markdown: The markdown content to scan
            
        Returns:
            List of AsciiTableInfo objects for detected tables
        """
        if not markdown or not markdown.strip():
            return []
        
        try:
            lines = markdown.split('\n')
            tables = []
            i = 0
            
            while i < len(lines):
                table_info = MarkdownTableProcessor._detect_table_at_position(lines, i)
                if table_info:
                    tables.append(table_info)
                    i = table_info.end_line + 1
                else:
                    i += 1
            
            logger.debug(f"Detected {len(tables)} ASCII tables in markdown")
            return tables
            
        except Exception as e:
            logger.error(f"Error detecting ASCII tables: {e}")
            return []
    
    @staticmethod
    def _detect_table_at_position(lines: List[str], start_idx: int) -> Optional[AsciiTableInfo]:
        """
        Try to detect a table starting at the given line position
        
        Args:
            lines: All lines in the document
            start_idx: Index to start checking from
            
        Returns:
            AsciiTableInfo if table found, None otherwise
        """
        if start_idx >= len(lines):
            return None
        
        # Check if current line looks like a table row
        current_line = lines[start_idx].strip()
        if not MarkdownTableProcessor._is_table_row(current_line):
            return None
        
        # Find the extent of the table
        table_start = start_idx
        table_end = start_idx
        
        # Scan forward to find all consecutive table rows
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if MarkdownTableProcessor._is_table_row(line) or MarkdownTableProcessor._is_divider_row(line):
                table_end = i
            else:
                break
        
        # Must have at least 2 lines to be a valid table
        if table_end - table_start < 1:
            return None
        
        # Extract table content and metadata
        table_lines = lines[table_start:table_end + 1]
        table_content = '\n'.join(table_lines)
        
        # Look for title in preceding lines
        title = MarkdownTableProcessor._find_table_title(lines, table_start)
        
        # Extract column information
        columns = MarkdownTableProcessor._extract_columns(table_lines)
        
        # Count data rows (excluding header and divider)
        row_count = MarkdownTableProcessor._count_data_rows(table_lines)
        
        return AsciiTableInfo(
            start_line=table_start,
            end_line=table_end,
            content=table_content,
            title=title,
            columns=columns,
            row_count=row_count,
            raw_lines=table_lines
        )
    
    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if a line looks like a table row"""
        if not line.strip():
            return False
        
        for pattern in MarkdownTableProcessor.TABLE_ROW_PATTERNS:
            if pattern.match(line):
                return True
        return False
    
    @staticmethod
    def _is_divider_row(line: str) -> bool:
        """Check if a line is a GFM table header divider"""
        return bool(MarkdownTableProcessor.DIVIDER_PATTERN.match(line))
    
    @staticmethod
    def _find_table_title(lines: List[str], table_start: int) -> Optional[str]:
        """
        Look for a potential table title in the lines before the table
        
        Args:
            lines: All document lines
            table_start: Index where table starts
            
        Returns:
            Title string if found, None otherwise
        """
        # Look up to 3 lines before the table for a title
        for i in range(table_start - 1, max(-1, table_start - 4), -1):
            if i < 0:
                break
                
            line = lines[i].strip()
            if not line:
                continue
            
            # Check for markdown header (prioritize closest header)
            header_match = re.match(r'^#+\s+(.+)$', line)
            if header_match:
                return header_match.group(1).strip()
            
            # Check for text that might be a title (simple heuristic)
            if len(line) < 100 and not MarkdownTableProcessor._is_table_row(line):
                # If the line contains words that suggest it's a title
                title_indicators = ['summary', 'table', 'list', 'overview', 'results', 'dependencies']
                if any(indicator in line.lower() for indicator in title_indicators):
                    return line.strip()
        
        return None
    
    @staticmethod
    def _extract_columns(table_lines: List[str]) -> List[str]:
        """
        Extract column names from table lines
        
        Args:
            table_lines: Lines that make up the table
            
        Returns:
            List of column names
        """
        if not table_lines:
            return []
        
        # Try to find header row (usually first non-divider row)
        header_row = None
        for line in table_lines:
            if MarkdownTableProcessor._is_table_row(line) and not MarkdownTableProcessor._is_divider_row(line):
                header_row = line
                break
        
        if not header_row:
            return []
        
        # Extract column content
        columns = []
        # Remove leading/trailing pipes and split
        content = header_row.strip()
        if content.startswith('|'):
            content = content[1:]
        if content.endswith('|'):
            content = content[:-1]
        
        # Split by pipes and clean up
        raw_columns = content.split('|')
        for col in raw_columns:
            cleaned = col.strip()
            if cleaned:  # Skip empty columns
                columns.append(cleaned)
        
        return columns
    
    @staticmethod
    def _count_data_rows(table_lines: List[str]) -> int:
        """
        Count the number of data rows in the table (excluding header and divider)
        
        Args:
            table_lines: Lines that make up the table
            
        Returns:
            Number of data rows
        """
        if not table_lines:
            return 0
        
        table_rows = [line for line in table_lines if MarkdownTableProcessor._is_table_row(line)]
        divider_rows = [line for line in table_lines if MarkdownTableProcessor._is_divider_row(line)]
        
        # If there's a divider, count rows after it
        if divider_rows:
            divider_found = False
            data_rows = 0
            for line in table_lines:
                if MarkdownTableProcessor._is_divider_row(line):
                    divider_found = True
                    continue
                elif divider_found and MarkdownTableProcessor._is_table_row(line):
                    data_rows += 1
            return data_rows
        else:
            # No divider - assume first row is header, rest are data
            return max(0, len(table_rows) - 1)
    
    @staticmethod
    def remove_ascii_table(markdown: str, table_info: AsciiTableInfo) -> str:
        """
        Remove a specific ASCII table from markdown content
        
        Args:
            markdown: Original markdown content
            table_info: Information about the table to remove
            
        Returns:
            Markdown with the specified table removed
        """
        if not markdown or not table_info:
            return markdown
        
        try:
            lines = markdown.split('\n')
            
            # Validate table position
            if (table_info.start_line < 0 or 
                table_info.end_line >= len(lines) or 
                table_info.start_line > table_info.end_line):
                logger.warning(f"Invalid table position: {table_info.start_line}-{table_info.end_line}")
                return markdown
            
            # Remove the table lines
            before_table = lines[:table_info.start_line]
            after_table = lines[table_info.end_line + 1:]
            
            # Preserve spacing - add a single blank line where table was
            result_lines = before_table + [''] + after_table
            
            # Clean up excessive blank lines (no more than 2 consecutive)
            cleaned_lines = MarkdownTableProcessor._clean_excessive_whitespace(result_lines)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Error removing ASCII table: {e}")
            return markdown  # Return original on error
    
    @staticmethod
    def _clean_excessive_whitespace(lines: List[str]) -> List[str]:
        """
        Clean up excessive blank lines while preserving document structure
        
        Args:
            lines: List of lines to clean
            
        Returns:
            Cleaned list of lines
        """
        if not lines:
            return lines
        
        cleaned = []
        consecutive_blanks = 0
        
        for line in lines:
            if line.strip() == '':
                consecutive_blanks += 1
                if consecutive_blanks <= 2:  # Allow up to 2 consecutive blank lines
                    cleaned.append(line)
            else:
                consecutive_blanks = 0
                cleaned.append(line)
        
        return cleaned
    
    @staticmethod
    def extract_table_metadata(table_info: AsciiTableInfo) -> TableMetadata:
        """
        Extract metadata from ASCII table for matching purposes
        
        Args:
            table_info: ASCII table information
            
        Returns:
            TableMetadata object with extracted information
        """
        if not table_info:
            return TableMetadata(
                title=None,
                column_names=[],
                column_count=0,
                row_count=0,
                data_sample=[]
            )
        
        # Extract data sample from first few rows
        data_sample = MarkdownTableProcessor._extract_data_sample(table_info.raw_lines)
        
        return TableMetadata(
            title=table_info.title,
            column_names=table_info.columns,
            column_count=len(table_info.columns),
            row_count=table_info.row_count,
            data_sample=data_sample
        )
    
    @staticmethod
    def _extract_data_sample(table_lines: List[str]) -> List[str]:
        """
        Extract sample data values from table for similarity matching
        
        Args:
            table_lines: Raw table lines
            
        Returns:
            List of sample data values
        """
        sample_data = []
        found_divider = False
        sample_count = 0
        max_samples = 10  # Limit sample size
        
        for line in table_lines:
            if MarkdownTableProcessor._is_divider_row(line):
                found_divider = True
                continue
            
            if MarkdownTableProcessor._is_table_row(line):
                if found_divider or (not found_divider and len(sample_data) > 0):
                    # This is a data row
                    content = line.strip()
                    if content.startswith('|'):
                        content = content[1:]
                    if content.endswith('|'):
                        content = content[:-1]
                    
                    # Extract cell values
                    cells = [cell.strip() for cell in content.split('|')]
                    sample_data.extend(cells[:3])  # Take first 3 cells from each row
                    
                    sample_count += 1
                    if sample_count >= 3:  # Limit to first 3 data rows
                        break
        
        return sample_data[:max_samples]