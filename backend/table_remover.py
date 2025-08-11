#!/usr/bin/env python3
"""
Table remover for safely removing ASCII tables from markdown
"""

from typing import List
from table_matcher import TableMatch

class TableRemover:
    """Safely removes ASCII tables from markdown while preserving structure"""
    
    def remove_duplicate_tables(self, markdown: str, matches: List[TableMatch]) -> str:
        """
        Remove ASCII tables that were identified as duplicates
        
        Args:
            markdown: Original markdown content
            matches: List of TableMatch objects identifying duplicates
            
        Returns:
            Cleaned markdown with duplicate tables removed
        """
        if not matches:
            return markdown
        
        lines = markdown.split('\n')
        
        # Sort matches by start line in reverse order to avoid index shifting
        sorted_matches = sorted(matches, key=lambda m: m.ascii_start, reverse=True)
        
        # Remove each matched table
        for match in sorted_matches:
            # Remove lines from start to end (inclusive)
            del lines[match.ascii_start:match.ascii_end + 1]
        
        # Clean up any extra blank lines that might be left
        cleaned_lines = self._clean_extra_blank_lines(lines)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_extra_blank_lines(self, lines: List[str]) -> List[str]:
        """Remove excessive blank lines while preserving document structure"""
        cleaned = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                # Keep at most 1 consecutive blank line
                if blank_count <= 1:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)
        
        # Remove trailing blank lines
        while cleaned and cleaned[-1].strip() == '':
            cleaned.pop()
        
        return cleaned