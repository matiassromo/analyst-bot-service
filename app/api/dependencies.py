"""
FastAPI dependency injection container.
Provides services to route handlers.
"""

import logging
from functools import lru_cache

from app.core.config import settings
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMService
from app.services.chart_service import ChartService
from app.repositories.database_repository import DatabaseRepository
from app.utils.query_validator import QueryValidator

logger = logging.getLogger(__name__)


# Singleton instances (created once per application lifecycle)
_db_repository: DatabaseRepository = None
_llm_service: LLMService = None
_chart_service: ChartService = None
_query_validator: QueryValidator = None


def get_db_repository() -> DatabaseRepository:
    """
    Get database repository instance (singleton).

    Returns:
        DatabaseRepository instance
    """
    global _db_repository

    if _db_repository is None:
        logger.info("Initializing database repository...")
        _db_repository = DatabaseRepository(
            connection_string=settings.db_connection_string,
            max_rows=settings.max_query_rows
        )

    return _db_repository


def get_llm_service() -> LLMService:
    """
    Get LLM service instance (singleton).

    Returns:
        LLMService instance
    """
    global _llm_service

    if _llm_service is None:
        logger.info("Initializing LLM service...")
        _llm_service = LLMService(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model
        )

    return _llm_service


def get_chart_service() -> ChartService:
    """
    Get chart service instance (singleton).

    Returns:
        ChartService instance
    """
    global _chart_service

    if _chart_service is None:
        logger.info("Initializing chart service...")
        _chart_service = ChartService()

    return _chart_service


def get_query_validator() -> QueryValidator:
    """
    Get query validator instance (singleton).

    Returns:
        QueryValidator instance
    """
    global _query_validator

    if _query_validator is None:
        logger.info("Initializing query validator...")
        _query_validator = QueryValidator(max_rows=settings.max_query_rows)

    return _query_validator


def get_analysis_service() -> AnalysisService:
    """
    Get analysis service instance with dependencies.
    Creates a new instance for each request (analysis service is stateless).

    Returns:
        AnalysisService instance
    """
    return AnalysisService(
        llm_service=get_llm_service(),
        chart_service=get_chart_service(),
        db_repository=get_db_repository(),
        query_validator=get_query_validator()
    )


def reset_dependencies():
    """
    Reset all dependency singletons.
    Useful for testing or reloading configuration.
    """
    global _db_repository, _llm_service, _chart_service, _query_validator

    logger.info("Resetting all dependency singletons...")

    _db_repository = None
    _llm_service = None
    _chart_service = None
    _query_validator = None

    logger.info("Dependencies reset complete")
