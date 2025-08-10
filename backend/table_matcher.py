#!/usr/bin/env python3

import re
from typing import List, Tuple

class TableMatch:
    def __init__(self, ascii_start: int, ascii_end: int, confidence: float, reasons: List[str]):
        self.ascii_start = ascii_start
        self.ascii_end = ascii_end
        self.confidence = confidence
        self.reasons = reasons

class TableMatcher:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def find_duplicate_tables(self, markdown: str, structured_tables: List[dict]) -> List[TableMatch]:
        matches = []
        ascii_tables = self._find_ascii_tables(markdown)
        
        for ascii_table in ascii_tables:
            for structured_table in structured_tables:
                confidence, reasons = self._calculate_similarity(ascii_table, structured_table)
                
                if confidence >= self.threshold:
                    matches.append(TableMatch(
                        ascii_start=ascii_table['start'],
                        ascii_end=ascii_table['end'],
                        confidence=confidence,
                        reasons=reasons
                    ))
                    break
        
        return matches
    
    def _find_ascii_tables(self, markdown: str) -> List[dict]:
        lines = markdown.split('\n')
        tables = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if '|' in line and len(line.split('|')) >= 3:
                start_idx = i
                table_lines = [line]
                i += 1
                
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i].strip())
                    i += 1
                
                if len(table_lines) >= 2:
                    tables.append({
                        'start': start_idx,
                        'end': i - 1,
                        'lines': table_lines,
                        'columns': len([cell for cell in table_lines[0].split('|') if cell.strip()]),
                        'rows': len(table_lines) - 1
                    })
            else:
                i += 1
        
        return tables
    
    def _calculate_similarity(self, ascii_table: dict, structured_table: dict) -> Tuple[float, List[str]]:
        reasons = []
        score = 0.0
        
        ascii_cols = ascii_table['columns']
        struct_cols = len(structured_table.get('columns', []))
        
        if ascii_cols == struct_cols:
            score += 0.4
            reasons.append(f"Column count match: {ascii_cols}")
        
        ascii_rows = ascii_table['rows']
        struct_rows = len(structured_table.get('rows', []))
        
        if ascii_rows == struct_rows:
            score += 0.3
            reasons.append(f"Row count match: {ascii_rows}")
        
        score += 0.3  # Base similarity for having tables
        reasons.append("Both are tables")
        
        return score, reasons