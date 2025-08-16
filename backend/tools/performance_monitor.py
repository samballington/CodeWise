"""
Tool Performance Monitoring for Phase 1 Integration

Tracks how Phase 1 improvements affect existing tool performance and provides
comprehensive metrics for measuring search quality improvements.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolPerformanceStats:
    """Performance statistics for a specific tool."""
    tool_name: str
    total_calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    success_rate: float = 0.0
    recent_metrics: List[PerformanceMetric] = field(default_factory=list)


class ToolPerformanceMonitor:
    """
    Monitor Phase 1 integration impact on existing tools.
    
    Tracks performance improvements from BGE embeddings, hierarchical chunks,
    and semantic theming integration.
    """
    
    def __init__(self, storage_dir: str = "performance_logs"):
        """
        Initialize performance monitor.
        
        Args:
            storage_dir: Directory for storing performance logs
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.tool_stats: Dict[str, ToolPerformanceStats] = {}
        self.metrics_history: List[PerformanceMetric] = []
        
        # Search quality tracking
        self.search_improvements = {
            'baseline_scores': {},
            'enhanced_scores': {},
            'improvement_tracking': []
        }
        
        # Context quality tracking
        self.context_quality_metrics = {
            'examine_files_enhancements': [],
            'hierarchical_context_usage': [],
            'mermaid_theming_success': []
        }
        
        logger.info(f"Tool performance monitor initialized, storage: {storage_dir}")
    
    def start_tool_timing(self, tool_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start timing a tool operation.
        
        Args:
            tool_name: Name of the tool being timed
            metadata: Additional metadata for the operation
            
        Returns:
            Timing ID for stopping the timer
        """
        timing_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolPerformanceStats(tool_name=tool_name)
        
        # Store start time and metadata
        setattr(self, f"_start_{timing_id}", {
            'start_time': time.time(),
            'metadata': metadata or {}
        })
        
        return timing_id
    
    def end_tool_timing(self, timing_id: str, success: bool = True, 
                       result_metadata: Dict[str, Any] = None) -> float:
        """
        End timing for a tool operation.
        
        Args:
            timing_id: Timing ID from start_tool_timing
            success: Whether the operation was successful
            result_metadata: Additional result metadata
            
        Returns:
            Duration in seconds
        """
        start_data = getattr(self, f"_start_{timing_id}", None)
        if not start_data:
            logger.warning(f"No start time found for timing ID: {timing_id}")
            return 0.0
        
        duration = time.time() - start_data['start_time']
        # Extract tool name from timing_id (format: tool_name_timestamp)
        tool_name = timing_id.rsplit('_', 1)[0]
        
        # Update tool statistics
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolPerformanceStats(tool_name=tool_name)
        stats = self.tool_stats[tool_name]
        stats.total_calls += 1
        
        if success:
            stats.total_time += duration
            stats.avg_time = stats.total_time / stats.total_calls
            stats.min_time = min(stats.min_time, duration)
            stats.max_time = max(stats.max_time, duration)
        
        # Update success rate
        success_count = sum(1 for m in stats.recent_metrics 
                          if m.metadata.get('success', True))
        stats.success_rate = success_count / max(stats.total_calls, 1)
        
        # Create performance metric
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=f"{tool_name}_duration",
            value=duration,
            metadata={
                **start_data['metadata'],
                **(result_metadata or {}),
                'success': success
            }
        )
        
        stats.recent_metrics.append(metric)
        self.metrics_history.append(metric)
        
        # Clean up temporary start data
        delattr(self, f"_start_{timing_id}")
        
        logger.debug(f"Tool {tool_name} completed in {duration:.3f}s (success: {success})")
        return duration
    
    def track_search_improvement(self, query: str, old_results: List, new_results: List,
                                query_type: str = "general") -> None:
        """
        Track search quality improvements from BGE + hierarchical chunks.
        
        Args:
            query: Search query
            old_results: Results from old system
            new_results: Results from enhanced system
            query_type: Type of query for categorization
        """
        improvement_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_type': query_type,
            'old_result_count': len(old_results),
            'new_result_count': len(new_results),
            'old_avg_score': 0.0,
            'new_avg_score': 0.0,
            'improvement_percent': 0.0
        }
        
        # Calculate average relevance scores
        if old_results:
            old_scores = [getattr(r, 'relevance_score', 0.5) for r in old_results]
            improvement_data['old_avg_score'] = sum(old_scores) / len(old_scores)
        
        if new_results:
            new_scores = [getattr(r, 'relevance_score', 0.5) for r in new_results]
            improvement_data['new_avg_score'] = sum(new_scores) / len(new_scores)
        
        # Calculate improvement percentage
        if improvement_data['old_avg_score'] > 0:
            improvement = ((improvement_data['new_avg_score'] - improvement_data['old_avg_score']) / 
                          improvement_data['old_avg_score']) * 100
            improvement_data['improvement_percent'] = improvement
        
        self.search_improvements['improvement_tracking'].append(improvement_data)
        
        logger.debug(f"Search improvement tracked: {improvement_data['improvement_percent']:.2f}%")
    
    def track_examine_files_enhancement(self, file_path: str, enhancement_type: str,
                                       quality_score: float, metadata: Dict[str, Any] = None):
        """
        Track examine_files enhancements from hierarchical context.
        
        Args:
            file_path: Path of examined file
            enhancement_type: Type of enhancement (hierarchical_context, symbol_analysis, etc.)
            quality_score: Quality score (0-1)
            metadata: Additional metadata
        """
        enhancement_data = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'enhancement_type': enhancement_type,
            'quality_score': quality_score,
            'metadata': metadata or {}
        }
        
        self.context_quality_metrics['examine_files_enhancements'].append(enhancement_data)
        logger.debug(f"Examine files enhancement tracked: {enhancement_type} (score: {quality_score:.2f})")
    
    def track_mermaid_theming_success(self, diagram_type: str, theme_used: str,
                                     semantic_roles_assigned: int, success: bool):
        """
        Track mermaid theming integration success.
        
        Args:
            diagram_type: Type of diagram generated
            theme_used: Theme applied
            semantic_roles_assigned: Number of semantic roles assigned
            success: Whether theming was successful
        """
        theming_data = {
            'timestamp': datetime.now().isoformat(),
            'diagram_type': diagram_type,
            'theme_used': theme_used,
            'semantic_roles_assigned': semantic_roles_assigned,
            'success': success
        }
        
        self.context_quality_metrics['mermaid_theming_success'].append(theming_data)
        logger.debug(f"Mermaid theming tracked: {theme_used} (success: {success})")
    
    def track_hierarchical_context_usage(self, context_type: str, chunks_analyzed: int,
                                        relationships_found: int, effectiveness_score: float):
        """
        Track hierarchical context usage effectiveness.
        
        Args:
            context_type: Type of context analysis
            chunks_analyzed: Number of chunks analyzed
            relationships_found: Number of relationships found
            effectiveness_score: Effectiveness score (0-1)
        """
        context_data = {
            'timestamp': datetime.now().isoformat(),
            'context_type': context_type,
            'chunks_analyzed': chunks_analyzed,
            'relationships_found': relationships_found,
            'effectiveness_score': effectiveness_score
        }
        
        self.context_quality_metrics['hierarchical_context_usage'].append(context_data)
        logger.debug(f"Hierarchical context usage tracked: {context_type} (score: {effectiveness_score:.2f})")
    
    def get_performance_report(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive performance impact report.
        
        Args:
            days_back: Number of days to include in report
            
        Returns:
            Performance report dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'period_days': days_back,
            'tool_performance': {},
            'search_improvements': {},
            'context_quality': {},
            'overall_impact': {}
        }
        
        # Tool performance summary
        for tool_name, stats in self.tool_stats.items():
            recent_metrics = [m for m in stats.recent_metrics 
                            if m.timestamp >= cutoff_date]
            
            if recent_metrics:
                recent_durations = [m.value for m in recent_metrics]
                report['tool_performance'][tool_name] = {
                    'total_calls': len(recent_metrics),
                    'avg_duration': sum(recent_durations) / len(recent_durations),
                    'min_duration': min(recent_durations),
                    'max_duration': max(recent_durations),
                    'success_rate': sum(1 for m in recent_metrics 
                                      if m.metadata.get('success', True)) / len(recent_metrics)
                }
        
        # Search improvements summary
        recent_improvements = [imp for imp in self.search_improvements['improvement_tracking']
                             if datetime.fromisoformat(imp['timestamp']) >= cutoff_date]
        
        if recent_improvements:
            improvements = [imp['improvement_percent'] for imp in recent_improvements]
            report['search_improvements'] = {
                'total_queries': len(recent_improvements),
                'avg_improvement_percent': sum(improvements) / len(improvements),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'max_improvement': max(improvements),
                'improvement_rate': sum(1 for imp in improvements if imp > 0) / len(improvements)
            }
        
        # Context quality summary
        examine_enhancements = [e for e in self.context_quality_metrics['examine_files_enhancements']
                               if datetime.fromisoformat(e['timestamp']) >= cutoff_date]
        
        if examine_enhancements:
            quality_scores = [e['quality_score'] for e in examine_enhancements]
            report['context_quality']['examine_files'] = {
                'total_enhancements': len(examine_enhancements),
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'high_quality_rate': sum(1 for score in quality_scores if score > 0.7) / len(quality_scores)
            }
        
        # Mermaid theming summary
        theming_data = [t for t in self.context_quality_metrics['mermaid_theming_success']
                       if datetime.fromisoformat(t['timestamp']) >= cutoff_date]
        
        if theming_data:
            report['context_quality']['mermaid_theming'] = {
                'total_diagrams': len(theming_data),
                'success_rate': sum(1 for t in theming_data if t['success']) / len(theming_data),
                'avg_semantic_roles': sum(t['semantic_roles_assigned'] for t in theming_data) / len(theming_data)
            }
        
        # Overall impact assessment
        report['overall_impact'] = self._calculate_overall_impact(report)
        
        return report
    
    def _calculate_overall_impact(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall impact metrics."""
        impact = {
            'search_quality_improvement': 0.0,
            'tool_performance_impact': 0.0,
            'context_enhancement_effectiveness': 0.0,
            'overall_score': 0.0
        }
        
        # Search quality improvement
        search_data = report.get('search_improvements', {})
        if 'avg_improvement_percent' in search_data:
            impact['search_quality_improvement'] = max(0, search_data['avg_improvement_percent'])
        
        # Tool performance impact (negative if tools got slower)
        tool_data = report.get('tool_performance', {})
        if tool_data:
            avg_success_rates = [data['success_rate'] for data in tool_data.values()]
            if avg_success_rates:
                impact['tool_performance_impact'] = sum(avg_success_rates) / len(avg_success_rates)
        
        # Context enhancement effectiveness
        context_data = report.get('context_quality', {})
        effectiveness_scores = []
        
        if 'examine_files' in context_data:
            effectiveness_scores.append(context_data['examine_files']['avg_quality_score'])
        
        if 'mermaid_theming' in context_data:
            effectiveness_scores.append(context_data['mermaid_theming']['success_rate'])
        
        if effectiveness_scores:
            impact['context_enhancement_effectiveness'] = sum(effectiveness_scores) / len(effectiveness_scores)
        
        # Overall score (weighted average)
        weights = [0.4, 0.3, 0.3]  # Search quality, tool performance, context enhancement
        scores = [
            impact['search_quality_improvement'] / 100,  # Normalize percentage
            impact['tool_performance_impact'],
            impact['context_enhancement_effectiveness']
        ]
        
        if any(score > 0 for score in scores):
            impact['overall_score'] = sum(w * s for w, s in zip(weights, scores) if s > 0)
        
        return impact
    
    def save_performance_data(self) -> None:
        """Save performance data to files."""
        try:
            # Save tool statistics
            stats_file = self.storage_dir / "tool_stats.json"
            with open(stats_file, 'w') as f:
                stats_data = {}
                for tool_name, stats in self.tool_stats.items():
                    stats_data[tool_name] = {
                        'total_calls': stats.total_calls,
                        'total_time': stats.total_time,
                        'avg_time': stats.avg_time,
                        'min_time': stats.min_time if stats.min_time != float('inf') else 0,
                        'max_time': stats.max_time,
                        'success_rate': stats.success_rate
                    }
                json.dump(stats_data, f, indent=2)
            
            # Save metrics history
            metrics_file = self.storage_dir / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                metrics_data = []
                for metric in self.metrics_history[-1000:]:  # Keep last 1000 metrics
                    metrics_data.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'metric_name': metric.metric_name,
                        'value': metric.value,
                        'metadata': metric.metadata
                    })
                json.dump(metrics_data, f, indent=2)
            
            # Save improvement tracking
            improvements_file = self.storage_dir / "search_improvements.json"
            with open(improvements_file, 'w') as f:
                json.dump(self.search_improvements, f, indent=2)
            
            logger.info(f"Performance data saved to {self.storage_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def load_performance_data(self) -> bool:
        """
        Load performance data from files.
        
        Returns:
            True if data loaded successfully
        """
        try:
            # Load tool statistics
            stats_file = self.storage_dir / "tool_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                
                for tool_name, data in stats_data.items():
                    stats = ToolPerformanceStats(tool_name=tool_name)
                    stats.total_calls = data['total_calls']
                    stats.total_time = data['total_time']
                    stats.avg_time = data['avg_time']
                    stats.min_time = data['min_time']
                    stats.max_time = data['max_time']
                    stats.success_rate = data['success_rate']
                    self.tool_stats[tool_name] = stats
            
            # Load improvement tracking
            improvements_file = self.storage_dir / "search_improvements.json"
            if improvements_file.exists():
                with open(improvements_file, 'r') as f:
                    self.search_improvements = json.load(f)
            
            logger.info("Performance data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            return False
    
    def clear_old_data(self, days_to_keep: int = 30) -> None:
        """Clear performance data older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clear old metrics
        self.metrics_history = [m for m in self.metrics_history 
                               if m.timestamp >= cutoff_date]
        
        # Clear old improvement tracking
        self.search_improvements['improvement_tracking'] = [
            imp for imp in self.search_improvements['improvement_tracking']
            if datetime.fromisoformat(imp['timestamp']) >= cutoff_date
        ]
        
        # Clear old context metrics
        for metric_list in self.context_quality_metrics.values():
            if isinstance(metric_list, list):
                metric_list[:] = [
                    item for item in metric_list
                    if datetime.fromisoformat(item['timestamp']) >= cutoff_date
                ]
        
        logger.info(f"Cleared performance data older than {days_to_keep} days")


# Global performance monitor instance
performance_monitor = ToolPerformanceMonitor()