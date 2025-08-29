"""
Benchmarking and Evaluation Framework for CodeWise

This module provides comprehensive evaluation tools for measuring
search quality improvements from Phase 1 enhancements.
"""

from .evaluator import RetrievalEvaluator
from .migration_manager import ModelMigrationManager

__all__ = ['RetrievalEvaluator', 'ModelMigrationManager']