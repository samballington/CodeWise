"""
Backend Services Package

REQ-3.7.1: Startup services for automatic Knowledge Graph population
"""

from .kg_startup_service import (
    KGStartupService,
    KGStartupResult, 
    ProjectIndexingResult,
    get_kg_startup_service,
    run_startup_kg_population
)

__all__ = [
    "KGStartupService",
    "KGStartupResult", 
    "ProjectIndexingResult",
    "get_kg_startup_service",
    "run_startup_kg_population"
]