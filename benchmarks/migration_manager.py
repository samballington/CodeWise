"""
Model Migration Manager

Manages migration from old embedding model to new BGE model with
validation and performance comparison.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.evaluator import RetrievalEvaluator
from indexer.enhanced_vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)


class ModelMigrationManager:
    """
    Manages migration from old embedding model to new BGE model.
    
    Provides comprehensive migration with before/after validation,
    performance comparison, and rollback capabilities.
    """
    
    def __init__(self, old_vector_store, new_vector_store: EnhancedVectorStore, 
                 evaluator: RetrievalEvaluator):
        """
        Initialize migration manager.
        
        Args:
            old_vector_store: Current vector store (MiniLM-based)
            new_vector_store: New BGE-based vector store
            evaluator: Evaluation framework for comparison
        """
        self.old_store = old_vector_store
        self.new_store = new_vector_store
        self.evaluator = evaluator
        
        # Migration tracking
        self.migration_log = []
        self.backup_created = False
        self.migration_completed = False
        
        logger.info("ModelMigrationManager initialized")
    
    def migrate_with_validation(self, chunks: List[str], chunk_metadata: List[Dict] = None,
                               backup_old: bool = True) -> Dict[str, Any]:
        """
        Perform complete migration with before/after validation.
        
        Args:
            chunks: List of text chunks to re-embed
            chunk_metadata: Optional metadata for chunks
            backup_old: Whether to backup old vector store
            
        Returns:
            Migration report with performance comparison
        """
        migration_start = time.time()
        logger.info("Starting embedding model migration with validation...")
        
        try:
            # Step 1: Backup old vector store
            if backup_old:
                self._create_backup()
            
            # Step 2: Evaluate old model performance
            logger.info("Evaluating current model performance...")
            old_results = self._evaluate_old_model()
            
            # Step 3: Build new index with BGE model
            logger.info(f"Building new BGE index for {len(chunks)} chunks...")
            build_success = self._build_new_index(chunks, chunk_metadata)
            
            if not build_success:
                raise Exception("Failed to build new BGE index")
            
            # Step 4: Evaluate new model performance
            logger.info("Evaluating new BGE model performance...")
            new_results = self._evaluate_new_model()
            
            # Step 5: Compare performance
            improvements = self.evaluator.compare_models(old_results, new_results)
            
            # Step 6: Validate migration success
            migration_valid = self._validate_migration(old_results, new_results, improvements)
            
            # Step 7: Generate migration report
            migration_time = time.time() - migration_start
            migration_report = self._generate_migration_report(
                old_results, new_results, improvements, migration_time, 
                len(chunks), migration_valid
            )
            
            if migration_valid:
                self.migration_completed = True
                logger.info("Migration completed successfully!")
            else:
                logger.warning("Migration completed with validation warnings")
            
            return migration_report
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if backup_old and self.backup_created:
                logger.info("Consider restoring from backup if needed")
            
            return {
                'migration_status': 'failed',
                'error': str(e),
                'migration_time': time.time() - migration_start,
                'backup_created': self.backup_created
            }
    
    def rollback_migration(self) -> bool:
        """
        Rollback to old vector store if backup exists.
        
        Returns:
            True if rollback successful
        """
        if not self.backup_created:
            logger.error("No backup available for rollback")
            return False
        
        try:
            backup_dir = Path(".vector_cache_backup")
            if not backup_dir.exists():
                logger.error("Backup directory not found")
                return False
            
            # Restore old files
            original_dir = Path(".vector_cache")
            
            # Remove new files
            if original_dir.exists():
                import shutil
                shutil.rmtree(original_dir)
            
            # Restore backup
            shutil.copytree(backup_dir, original_dir)
            
            logger.info("Successfully rolled back to old vector store")
            self.migration_completed = False
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status.
        
        Returns:
            Dictionary with migration status information
        """
        return {
            'migration_completed': self.migration_completed,
            'backup_created': self.backup_created,
            'old_store_type': type(self.old_store).__name__,
            'new_store_type': type(self.new_store).__name__,
            'migration_log_entries': len(self.migration_log)
        }
    
    def _create_backup(self) -> None:
        """Create backup of old vector store."""
        try:
            import shutil
            
            source_dir = Path(".vector_cache")
            backup_dir = Path(".vector_cache_backup")
            
            if source_dir.exists():
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                
                shutil.copytree(source_dir, backup_dir)
                self.backup_created = True
                logger.info(f"Created backup at {backup_dir}")
                
                self._log_migration_step("backup_created", "Old vector store backed up")
            else:
                logger.warning("No existing vector cache to backup")
                
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _evaluate_old_model(self) -> Dict[str, float]:
        """Evaluate old model performance."""
        try:
            # Create a wrapper for the old vector store to match evaluator interface
            old_search_engine = OldStoreWrapper(self.old_store)
            results = self.evaluator.evaluate_model(old_search_engine, save_results=False)
            
            self._log_migration_step("old_model_evaluated", f"P@3: {results['precision_at_3']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Old model evaluation failed: {e}")
            # Return default metrics to allow migration to continue
            return {
                'precision_at_3': 0.0,
                'precision_at_5': 0.0,
                'recall_at_10': 0.0,
                'mean_reciprocal_rank': 0.0
            }
    
    def _build_new_index(self, chunks: List[str], chunk_metadata: List[Dict] = None) -> bool:
        """Build new BGE-based index."""
        try:
            # Clear any existing index
            self._clear_new_vector_cache()
            
            # Build new index
            success = self.new_store.build_index(chunks, chunk_metadata)
            
            if success:
                self._log_migration_step("new_index_built", f"Built index with {len(chunks)} chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"New index building failed: {e}")
            return False
    
    def _evaluate_new_model(self) -> Dict[str, float]:
        """Evaluate new BGE model performance."""
        try:
            # Create wrapper for new vector store
            new_search_engine = NewStoreWrapper(self.new_store)
            results = self.evaluator.evaluate_model(new_search_engine, save_results=False)
            
            self._log_migration_step("new_model_evaluated", f"P@3: {results['precision_at_3']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"New model evaluation failed: {e}")
            # Return default metrics
            return {
                'precision_at_3': 0.0,
                'precision_at_5': 0.0,
                'recall_at_10': 0.0,
                'mean_reciprocal_rank': 0.0
            }
    
    def _validate_migration(self, old_results: Dict, new_results: Dict, 
                           improvements: Dict) -> bool:
        """Validate that migration was successful."""
        validation_checks = []
        
        # Check 1: New model should perform at least as well as old model
        main_metrics = ['precision_at_3', 'precision_at_5', 'recall_at_10', 'mean_reciprocal_rank']
        
        for metric in main_metrics:
            old_score = old_results.get(metric, 0)
            new_score = new_results.get(metric, 0)
            
            if new_score >= old_score:
                validation_checks.append(f"✓ {metric}: improved from {old_score:.3f} to {new_score:.3f}")
            else:
                validation_checks.append(f"⚠ {metric}: decreased from {old_score:.3f} to {new_score:.3f}")
        
        # Check 2: At least some metrics should show improvement
        improved_metrics = sum(1 for metric in main_metrics 
                             if new_results.get(metric, 0) > old_results.get(metric, 0))
        
        if improved_metrics >= len(main_metrics) // 2:
            validation_checks.append(f"✓ {improved_metrics}/{len(main_metrics)} metrics improved")
        else:
            validation_checks.append(f"⚠ Only {improved_metrics}/{len(main_metrics)} metrics improved")
        
        # Check 3: Overall improvement should be positive
        overall_improvement = improvements.get('overall_improvement_percent', 0)
        if overall_improvement > 0:
            validation_checks.append(f"✓ Overall improvement: {overall_improvement:.2f}%")
        else:
            validation_checks.append(f"⚠ Overall improvement: {overall_improvement:.2f}%")
        
        # Log validation results
        for check in validation_checks:
            logger.info(f"Migration validation: {check}")
        
        # Migration is valid if more than half the checks pass
        passed_checks = sum(1 for check in validation_checks if check.startswith("✓"))
        return passed_checks > len(validation_checks) // 2
    
    def _generate_migration_report(self, old_results: Dict, new_results: Dict, 
                                  improvements: Dict, migration_time: float,
                                  chunks_processed: int, migration_valid: bool) -> Dict[str, Any]:
        """Generate comprehensive migration report."""
        report = {
            'migration_metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'migration_time_seconds': migration_time,
                'chunks_reindexed': chunks_processed,
                'migration_valid': migration_valid,
                'migration_status': 'completed' if migration_valid else 'completed_with_warnings'
            },
            'old_model_results': old_results,
            'new_model_results': new_results,
            'improvements': improvements,
            'performance_summary': {
                'precision_at_3_change': new_results.get('precision_at_3', 0) - old_results.get('precision_at_3', 0),
                'recall_at_10_change': new_results.get('recall_at_10', 0) - old_results.get('recall_at_10', 0),
                'mrr_change': new_results.get('mean_reciprocal_rank', 0) - old_results.get('mean_reciprocal_rank', 0)
            },
            'migration_log': self.migration_log,
            'model_info': {
                'old_model': self._get_old_model_info(),
                'new_model': self.new_store.get_model_info()
            }
        }
        
        # Save report
        self._save_migration_report(report)
        
        # Log summary
        avg_improvement = improvements.get('overall_improvement_percent', 0)
        logger.info(f"Migration completed in {migration_time:.2f}s with {avg_improvement:.2f}% average improvement")
        
        return report
    
    def _clear_new_vector_cache(self) -> None:
        """Clear new vector cache directory."""
        try:
            cache_dir = Path(".vector_cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            logger.warning(f"Failed to clear vector cache: {e}")
    
    def _log_migration_step(self, step: str, details: str) -> None:
        """Log migration step for tracking."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'step': step,
            'details': details
        }
        self.migration_log.append(log_entry)
        logger.debug(f"Migration step: {step} - {details}")
    
    def _get_old_model_info(self) -> Dict[str, Any]:
        """Get information about old model."""
        info = {'model_type': type(self.old_store).__name__}
        
        if hasattr(self.old_store, 'get_model_info'):
            try:
                info.update(self.old_store.get_model_info())
            except:
                pass
        else:
            # Default info for legacy vector stores
            info.update({
                'model_name': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384,
                'normalization_enabled': True
            })
        
        return info
    
    def _save_migration_report(self, report: Dict[str, Any]) -> None:
        """Save migration report to file."""
        try:
            reports_dir = Path("benchmarks/migration_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"migration_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Migration report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save migration report: {e}")


class OldStoreWrapper:
    """Wrapper to adapt old vector store for evaluation."""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def search(self, query: str, limit: int = 10):
        """Adapt old vector store search method."""
        try:
            if hasattr(self.vector_store, 'search'):
                return self.vector_store.search(query, limit=limit)
            elif hasattr(self.vector_store, 'query'):
                similarities, indices = self.vector_store.query(query, limit)
                # Convert to expected format
                results = []
                for i, (sim, idx) in enumerate(zip(similarities, indices)):
                    result = type('Result', (), {
                        'chunk_id': f"chunk_{idx}",
                        'relevance_score': float(sim)
                    })()
                    results.append(result)
                return results
            else:
                return []
        except Exception as e:
            logger.error(f"Old store search failed: {e}")
            return []


class NewStoreWrapper:
    """Wrapper to adapt new BGE vector store for evaluation."""
    
    def __init__(self, vector_store: EnhancedVectorStore):
        self.vector_store = vector_store
    
    def search(self, query: str, limit: int = 10):
        """Adapt new vector store search method."""
        try:
            similarities, indices = self.vector_store.search(query, limit)
            
            # Convert to expected format
            results = []
            for i, (sim, idx) in enumerate(zip(similarities, indices)):
                result = type('Result', (), {
                    'chunk_id': f"chunk_{idx}",
                    'relevance_score': float(sim)
                })()
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"New store search failed: {e}")
            return []