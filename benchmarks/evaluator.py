"""
Retrieval Evaluation Framework

Objective evaluation framework for measuring retrieval quality improvements
using golden set benchmarks and standard IR metrics.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result for evaluation."""
    chunk_id: str
    relevance_score: float
    rank: int
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    query_id: str
    query: str
    metrics: Dict[str, float]
    retrieved_chunks: List[str]
    expected_chunks: List[str]
    precision_at_k: Dict[int, float]
    found_at_ranks: List[int]  # Ranks where expected chunks were found
    

class RetrievalEvaluator:
    """
    Objective evaluation framework for measuring retrieval quality improvements.
    
    Compares search results against golden set ground truth using standard
    information retrieval metrics.
    """
    
    def __init__(self, golden_set_path: str = "benchmarks/golden_set.json"):
        """
        Initialize evaluator with golden set.
        
        Args:
            golden_set_path: Path to golden set benchmark file
        """
        self.golden_set_path = Path(golden_set_path)
        self.golden_set = self._load_golden_set()
        self.queries = self.golden_set.get('benchmark_queries', [])
        
        logger.info(f"Loaded {len(self.queries)} benchmark queries from {golden_set_path}")
    
    def evaluate_model(self, search_engine, top_k: int = 10, 
                      save_results: bool = True) -> Dict[str, float]:
        """
        Evaluate search engine performance against golden set.
        
        Args:
            search_engine: Search engine instance with search() method
            top_k: Number of results to evaluate
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation with top_k={top_k}")
        
        # Store individual query results
        query_results: List[EvaluationResult] = []
        
        # Aggregate metrics
        all_precision_at_3 = []
        all_precision_at_5 = []
        all_recall_at_10 = []
        all_mrr = []
        
        # Category-specific metrics
        category_metrics = {}
        
        for i, query_data in enumerate(self.queries):
            logger.debug(f"Evaluating query {i+1}/{len(self.queries)}: {query_data['query_id']}")
            
            query = query_data['query']
            expected_chunks = set(query_data['expected_chunk_ids'])
            category = query_data.get('category', 'unknown')
            
            try:
                # Get search results
                search_results = self._get_search_results(search_engine, query, top_k)
                result_chunks = [r.chunk_id for r in search_results]
                
                # Calculate metrics for this query
                metrics = self._calculate_query_metrics(result_chunks, expected_chunks, top_k)
                
                # Store results
                query_result = EvaluationResult(
                    query_id=query_data['query_id'],
                    query=query,
                    metrics=metrics,
                    retrieved_chunks=result_chunks,
                    expected_chunks=list(expected_chunks),
                    precision_at_k=metrics['precision_at_k'],
                    found_at_ranks=metrics['found_at_ranks']
                )
                query_results.append(query_result)
                
                # Aggregate metrics
                all_precision_at_3.append(metrics['precision_at_3'])
                all_precision_at_5.append(metrics['precision_at_5'])
                all_recall_at_10.append(metrics['recall_at_10'])
                all_mrr.append(metrics['mrr'])
                
                # Category-specific aggregation
                if category not in category_metrics:
                    category_metrics[category] = {
                        'precision_at_3': [],
                        'precision_at_5': [],
                        'recall_at_10': [],
                        'mrr': []
                    }
                
                category_metrics[category]['precision_at_3'].append(metrics['precision_at_3'])
                category_metrics[category]['precision_at_5'].append(metrics['precision_at_5'])
                category_metrics[category]['recall_at_10'].append(metrics['recall_at_10'])
                category_metrics[category]['mrr'].append(metrics['mrr'])
                
            except Exception as e:
                logger.error(f"Error evaluating query {query_data['query_id']}: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = {
            'precision_at_3': np.mean(all_precision_at_3),
            'precision_at_5': np.mean(all_precision_at_5),
            'recall_at_10': np.mean(all_recall_at_10),
            'mean_reciprocal_rank': np.mean(all_mrr),
            'total_queries': len(query_results),
            'successful_queries': len([r for r in query_results if r.metrics['precision_at_3'] > 0])
        }
        
        # Add category breakdowns
        for category, metrics in category_metrics.items():
            overall_metrics[f'{category}_precision_at_3'] = np.mean(metrics['precision_at_3'])
            overall_metrics[f'{category}_precision_at_5'] = np.mean(metrics['precision_at_5'])
            overall_metrics[f'{category}_recall_at_10'] = np.mean(metrics['recall_at_10'])
            overall_metrics[f'{category}_mrr'] = np.mean(metrics['mrr'])
        
        # Save detailed results if requested
        if save_results:
            self._save_evaluation_results(overall_metrics, query_results, search_engine)
        
        logger.info(f"Evaluation completed: P@3={overall_metrics['precision_at_3']:.3f}, "
                   f"P@5={overall_metrics['precision_at_5']:.3f}, "
                   f"R@10={overall_metrics['recall_at_10']:.3f}, "
                   f"MRR={overall_metrics['mean_reciprocal_rank']:.3f}")
        
        return overall_metrics
    
    def compare_models(self, old_results: Dict[str, float], 
                      new_results: Dict[str, float]) -> Dict[str, float]:
        """
        Compare two model evaluation results.
        
        Args:
            old_results: Results from baseline model
            new_results: Results from new model
            
        Returns:
            Dictionary with improvement percentages
        """
        improvements = {}
        
        # Calculate improvements for main metrics
        main_metrics = ['precision_at_3', 'precision_at_5', 'recall_at_10', 'mean_reciprocal_rank']
        
        for metric in main_metrics:
            if metric in old_results and metric in new_results:
                old_score = old_results[metric]
                new_score = new_results[metric]
                
                if old_score > 0:
                    improvement = ((new_score - old_score) / old_score) * 100
                    improvements[f"{metric}_improvement_percent"] = improvement
                    improvements[f"{metric}_absolute_change"] = new_score - old_score
        
        # Calculate overall performance change
        if improvements:
            avg_improvement = np.mean([
                improvements.get('precision_at_3_improvement_percent', 0),
                improvements.get('recall_at_10_improvement_percent', 0),
                improvements.get('mean_reciprocal_rank_improvement_percent', 0)
            ])
            improvements['overall_improvement_percent'] = avg_improvement
        
        return improvements
    
    def evaluate_by_category(self, search_engine, categories: List[str] = None) -> Dict[str, Dict]:
        """
        Evaluate performance by query category.
        
        Args:
            search_engine: Search engine to evaluate
            categories: Specific categories to evaluate (None for all)
            
        Returns:
            Dictionary with category-specific results
        """
        if categories is None:
            categories = list(set(q.get('category', 'unknown') for q in self.queries))
        
        category_results = {}
        
        for category in categories:
            category_queries = [q for q in self.queries if q.get('category') == category]
            
            if not category_queries:
                logger.warning(f"No queries found for category: {category}")
                continue
            
            logger.info(f"Evaluating {len(category_queries)} queries in category: {category}")
            
            # Create temporary evaluator with only this category's queries
            temp_evaluator = RetrievalEvaluator.__new__(RetrievalEvaluator)
            temp_evaluator.golden_set = self.golden_set
            temp_evaluator.queries = category_queries
            
            # Evaluate
            results = temp_evaluator.evaluate_model(search_engine, save_results=False)
            category_results[category] = results
        
        return category_results
    
    def _load_golden_set(self) -> Dict:
        """Load golden set from file."""
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set not found: {self.golden_set_path}")
        
        try:
            with open(self.golden_set_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load golden set: {e}")
    
    def _get_search_results(self, search_engine, query: str, top_k: int) -> List[SearchResult]:
        """Get search results from search engine."""
        try:
            # Try different search engine interfaces
            if hasattr(search_engine, 'search'):
                # SmartSearchEngine or similar
                results = search_engine.search(query, limit=top_k)
                
                # Convert to SearchResult objects
                search_results = []
                for i, result in enumerate(results):
                    if hasattr(result, 'chunk_id'):
                        chunk_id = result.chunk_id
                    elif isinstance(result, dict):
                        chunk_id = result.get('chunk_id', f"result_{i}")
                    else:
                        chunk_id = f"result_{i}"
                    
                    relevance_score = getattr(result, 'relevance_score', 1.0 - i * 0.1)
                    
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        relevance_score=relevance_score,
                        rank=i + 1,
                        metadata=getattr(result, 'metadata', {})
                    ))
                
                return search_results
                
            elif hasattr(search_engine, 'query'):
                # Vector store interface
                similarities, indices = search_engine.query(query, top_k)
                
                search_results = []
                for i, (sim, idx) in enumerate(zip(similarities, indices)):
                    search_results.append(SearchResult(
                        chunk_id=f"chunk_{idx}",
                        relevance_score=float(sim),
                        rank=i + 1
                    ))
                
                return search_results
            
            else:
                logger.error(f"Unknown search engine interface: {type(search_engine)}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get search results: {e}")
            return []
    
    def _calculate_query_metrics(self, retrieved: List[str], expected: set, 
                                top_k: int) -> Dict[str, Any]:
        """Calculate metrics for a single query."""
        metrics = {}
        
        # Precision at K
        precision_at_k = {}
        for k in [3, 5, 10]:
            if k <= len(retrieved):
                relevant_at_k = sum(1 for chunk in retrieved[:k] if chunk in expected)
                precision_at_k[k] = relevant_at_k / k
            else:
                precision_at_k[k] = 0.0
        
        metrics['precision_at_k'] = precision_at_k
        metrics['precision_at_3'] = precision_at_k.get(3, 0.0)
        metrics['precision_at_5'] = precision_at_k.get(5, 0.0)
        
        # Recall at 10
        if len(expected) > 0:
            relevant_at_10 = sum(1 for chunk in retrieved[:10] if chunk in expected)
            metrics['recall_at_10'] = relevant_at_10 / len(expected)
        else:
            metrics['recall_at_10'] = 0.0
        
        # Mean Reciprocal Rank
        mrr = 0.0
        found_at_ranks = []
        for i, chunk in enumerate(retrieved, 1):
            if chunk in expected:
                if mrr == 0.0:  # First relevant result
                    mrr = 1.0 / i
                found_at_ranks.append(i)
        
        metrics['mrr'] = mrr
        metrics['found_at_ranks'] = found_at_ranks
        
        # Additional metrics
        metrics['total_expected'] = len(expected)
        metrics['total_retrieved'] = len(retrieved)
        metrics['total_relevant_found'] = len(found_at_ranks)
        
        return metrics
    
    def _save_evaluation_results(self, overall_metrics: Dict, query_results: List[EvaluationResult],
                                search_engine) -> None:
        """Save detailed evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("benchmarks") / f"evaluation_results_{timestamp}.json"
        
        # Prepare data for serialization
        results_data = {
            'timestamp': timestamp,
            'overall_metrics': overall_metrics,
            'model_info': self._get_model_info(search_engine),
            'query_results': [
                {
                    'query_id': r.query_id,
                    'query': r.query,
                    'metrics': r.metrics,
                    'retrieved_chunks': r.retrieved_chunks,
                    'expected_chunks': r.expected_chunks,
                    'found_at_ranks': r.found_at_ranks
                }
                for r in query_results
            ]
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def _get_model_info(self, search_engine) -> Dict[str, Any]:
        """Extract model information from search engine."""
        model_info = {'engine_type': type(search_engine).__name__}
        
        # Try to get model details
        if hasattr(search_engine, 'get_model_info'):
            try:
                model_info.update(search_engine.get_model_info())
            except:
                pass
        
        if hasattr(search_engine, 'vector_store'):
            vector_store = search_engine.vector_store
            if hasattr(vector_store, 'get_model_info'):
                try:
                    model_info.update(vector_store.get_model_info())
                except:
                    pass
        
        return model_info
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark queries."""
        categories = {}
        difficulties = {}
        
        for query in self.queries:
            category = query.get('category', 'unknown')
            difficulty = query.get('difficulty', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        return {
            'total_queries': len(self.queries),
            'categories': categories,
            'difficulties': difficulties,
            'golden_set_path': str(self.golden_set_path)
        }