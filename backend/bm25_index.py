"""
BM25 Indexing System for Keyword-Based Search

This module provides BM25 (Best Matching 25) indexing and search capabilities
for exact keyword matching to complement vector similarity search.
"""

import math
import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 search"""
    chunk_id: int
    score: float
    file_path: str
    snippet: str
    matched_terms: List[str]


class BM25Index:
    """BM25 indexing and search implementation"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index with tuning parameters
        
        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls length normalization (typical: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Index data structures
        self.documents: List[Dict] = []  # Document metadata
        self.term_frequencies: List[Dict[str, int]] = []  # TF for each document
        self.document_frequencies: Dict[str, int] = defaultdict(int)  # DF for each term
        self.document_lengths: List[int] = []  # Length of each document
        self.average_document_length: float = 0.0
        self.vocabulary: Set[str] = set()
        
        # Inverted index: term -> set of document IDs containing the term
        self.inverted: Dict[str, Set[int]] = defaultdict(set)
        
        # Statistics
        self.total_documents = 0
        
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the BM25 index
        
        Args:
            documents: List of document dictionaries with keys:
                - 'id': unique identifier
                - 'text': document text content
                - 'file_path': path to source file
                - 'metadata': additional metadata
        """
        logger.info(f"Adding {len(documents)} documents to BM25 index")
        
        for doc in documents:
            self._add_document(doc)
        
        # Calculate average document length
        if self.document_lengths:
            self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)
        
        logger.info(f"BM25 index built: {self.total_documents} documents, {len(self.vocabulary)} unique terms")
    
    def _add_document(self, document: Dict) -> None:
        """Add a single document to the index"""
        doc_id = len(self.documents)
        text = document.get('text', '')
        
        # Tokenize and count terms
        terms = self._tokenize(text)
        term_counts = Counter(terms)
        
        # Store document metadata
        self.documents.append({
            'id': document.get('id', doc_id),
            'file_path': document.get('file_path', ''),
            'text': text,
            'metadata': document.get('metadata', {})
        })
        
        # Store term frequencies for this document
        self.term_frequencies.append(dict(term_counts))
        
        # Update document frequencies (number of documents containing each term)
        unique_terms = set(terms)
        for term in unique_terms:
            self.document_frequencies[term] += 1
            # Add document to inverted index for this term
            self.inverted[term].add(doc_id)
        
        # Store document length and update vocabulary
        self.document_lengths.append(len(terms))
        self.vocabulary.update(unique_terms)
        self.total_documents += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for indexing
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of normalized terms
        """
        # Convert to lowercase and split on non-alphanumeric characters
        text = text.lower()
        
        # Extract words, keeping underscores and dots for technical terms
        terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\b', text)
        
        # Filter out very short terms and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        filtered_terms = []
        for term in terms:
            if len(term) >= 2 and term not in stop_words:
                filtered_terms.append(term)
        
        return filtered_terms
    
    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.1,
        allowed_projects: Optional[List[str]] = None,
    ) -> List[BM25Result]:
        """
        Search the index using BM25 scoring
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of BM25Result objects sorted by score (descending)
        """
        if not self.documents:
            logger.warning("BM25 index is empty")
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            logger.warning(f"No valid terms found in query: {query}")
            return []
        
        logger.debug(f"BM25 search for terms: {query_terms}")
        
        # Use inverted index to get candidate documents (union of docs containing any query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted:
                candidate_docs.update(self.inverted[term])
        
        # Fallback to full scan if no candidates found (shouldn't happen with valid terms)
        if not candidate_docs:
            logger.debug("No candidates found in inverted index, falling back to full scan")
            candidate_docs = set(range(len(self.documents)))
        
        logger.debug(f"BM25 candidate filtering: {len(candidate_docs)} candidates from {len(self.documents)} total docs")
        
        # Calculate BM25 scores for candidate documents only (major speedup!)
        scores = []
        for doc_id in candidate_docs:
            score = self._calculate_bm25_score(doc_id, query_terms)
            if score >= min_score:
                matched_terms = self._get_matched_terms(doc_id, query_terms)
                # Project scoping: drop docs outside allowed projects early
                if allowed_projects:
                    try:
                        file_path = self.documents[doc_id]['file_path']
                        project_dir = file_path.split('/')[0]
                        if project_dir not in set(allowed_projects):
                            continue
                    except Exception:
                        pass
                scores.append((doc_id, score, matched_terms))
        
        # Sort by score (descending) and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score, matched_terms in scores[:k]:
            doc = self.documents[doc_id]
            
            # Create snippet with highlighted terms
            snippet = self._create_snippet(doc['text'], matched_terms)
            
            result = BM25Result(
                chunk_id=doc_id,
                score=score,
                file_path=doc['file_path'],
                snippet=snippet,
                matched_terms=matched_terms
            )
            results.append(result)
        
        logger.info(f"BM25 search returned {len(results)} results for query: {query}")
        return results
    
    def _calculate_bm25_score(self, doc_id: int, query_terms: List[str]) -> float:
        """Calculate BM25 score for a document given query terms"""
        score = 0.0
        doc_length = self.document_lengths[doc_id]
        
        for term in query_terms:
            if term not in self.vocabulary:
                continue
            
            # Term frequency in document
            tf = self.term_frequencies[doc_id].get(term, 0)
            if tf == 0:
                continue
            
            # Document frequency (number of documents containing the term)
            df = self.document_frequencies[term]
            
            # Inverse document frequency
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def _get_matched_terms(self, doc_id: int, query_terms: List[str]) -> List[str]:
        """Get list of query terms that appear in the document"""
        matched = []
        doc_terms = set(self.term_frequencies[doc_id].keys())
        
        for term in query_terms:
            if term in doc_terms:
                matched.append(term)
        
        return matched
    
    def _create_snippet(self, text: str, matched_terms: List[str], max_length: int = 200) -> str:
        """Create a snippet of text highlighting matched terms"""
        if not matched_terms:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Find the best position to extract snippet (around first matched term)
        text_lower = text.lower()
        first_match_pos = len(text)
        
        for term in matched_terms:
            pos = text_lower.find(term.lower())
            if pos != -1 and pos < first_match_pos:
                first_match_pos = pos
        
        # Extract snippet around the first match
        start = max(0, first_match_pos - max_length // 2)
        end = min(len(text), start + max_length)
        
        snippet = text[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': self.average_document_length,
            'total_terms': sum(self.document_lengths),
            'parameters': {
                'k1': self.k1,
                'b': self.b
            }
        }
    
    def save_index(self, file_path: Path) -> bool:
        """Save the BM25 index to disk"""
        try:
            # Convert inverted index sets to lists for JSON serialization
            inverted_serializable = {
                term: list(doc_ids) for term, doc_ids in self.inverted.items()
            }
            
            index_data = {
                'documents': self.documents,
                'term_frequencies': self.term_frequencies,
                'document_frequencies': dict(self.document_frequencies),
                'document_lengths': self.document_lengths,
                'vocabulary': list(self.vocabulary),
                'inverted': inverted_serializable,
                'total_documents': self.total_documents,
                'average_document_length': self.average_document_length,
                'parameters': {'k1': self.k1, 'b': self.b}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"BM25 index saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
            return False
    
    def load_index(self, file_path: Path) -> bool:
        """Load the BM25 index from disk"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.documents = index_data['documents']
            self.term_frequencies = index_data['term_frequencies']
            self.document_frequencies = defaultdict(int, index_data['document_frequencies'])
            self.document_lengths = index_data['document_lengths']
            self.vocabulary = set(index_data['vocabulary'])
            self.total_documents = index_data['total_documents']
            self.average_document_length = index_data['average_document_length']
            
            # Load inverted index, converting lists back to sets
            inverted_data = index_data.get('inverted', {})
            self.inverted = defaultdict(set)
            for term, doc_ids in inverted_data.items():
                self.inverted[term] = set(doc_ids)
            
            # If no inverted index in saved data, rebuild it from term_frequencies
            if not inverted_data and self.term_frequencies:
                logger.info("Rebuilding inverted index from existing term frequencies")
                for doc_id, term_freqs in enumerate(self.term_frequencies):
                    for term in term_freqs.keys():
                        self.inverted[term].add(doc_id)
            
            # Load parameters
            params = index_data.get('parameters', {})
            self.k1 = params.get('k1', 1.5)
            self.b = params.get('b', 0.75)
            
            logger.info(f"BM25 index loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the index"""
        self.documents.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()
        self.vocabulary.clear()
        self.inverted.clear()
        self.total_documents = 0
        self.average_document_length = 0.0