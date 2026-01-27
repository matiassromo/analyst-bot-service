"""
Analysis service - Main orchestrator for the multi-query analysis workflow.
Coordinates LLM, database services for comprehensive data analysis.
"""

import logging
import time
from typing import List, Dict, Any

from app.services.llm_service import LLMService, LLMAPIError, LLMParseError
from app.repositories.database_repository import (
    DatabaseRepository,
    DatabaseConnectionError
)
from app.utils.query_validator import QueryValidator, QueryValidationError
from app.models.llm_models import QueryPlan

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base exception for analysis errors."""
    pass


class AnalysisService:
    """
    Main service orchestrator for multi-query data analysis workflow.

    Workflow:
    1. Get database schema
    2. Call LLM to generate multi-query plan (1-5 queries)
    3. Validate and execute each query
    4. Generate unified analysis from all results
    5. Return structured response with all query results and analysis
    """

    def __init__(
        self,
        llm_service: LLMService,
        db_repository: DatabaseRepository,
        query_validator: QueryValidator = None
    ):
        """
        Initialize analysis service with dependencies.

        Args:
            llm_service: LLM service for query generation
            db_repository: Database repository for query execution
            query_validator: Query validator (optional, creates new if not provided)
        """
        self.llm_service = llm_service
        self.db_repository = db_repository
        self.query_validator = query_validator or QueryValidator()

        logger.info("Analysis service initialized")

    async def analyze(
        self,
        user_prompt: str,
        exclude_tables: List[str] = None,
        max_queries: int = 5
    ) -> Dict[str, Any]:
        """
        Perform complete multi-query analysis workflow.

        Args:
            user_prompt: User's question in Spanish
            exclude_tables: Tables to exclude from schema (optional)
            max_queries: Maximum number of queries to generate (1-5)

        Returns:
            Dictionary with:
            - analysis: Unified analysis text in Spanish
            - queries: List of query results
            - metadata: Execution metadata

        Raises:
            AnalysisError: If any critical step in the workflow fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting multi-query analysis for prompt: {user_prompt[:100]}...")

            # Step 1: Get database schema
            logger.info("Step 1: Loading database schema...")
            try:
                schema = self.db_repository.get_schema(exclude_tables)
                logger.debug(f"Schema loaded: {len(schema)} characters")
            except DatabaseConnectionError as e:
                logger.error(f"Failed to load schema: {e}")
                raise AnalysisError(
                    "No se pudo conectar a la base de datos. "
                    "Por favor, verifica la configuracion."
                ) from e

            # Step 2: Call LLM for multi-query plan
            logger.info("Step 2: Generating multi-query plan with LLM...")
            try:
                llm_response = await self.llm_service.generate_multi_query_plan(
                    user_prompt=user_prompt,
                    database_schema=schema,
                    max_queries=max_queries
                )
                logger.info(f"LLM generated {len(llm_response.queries)} queries")
            except (LLMAPIError, LLMParseError) as e:
                logger.error(f"LLM generation failed: {e}")
                raise AnalysisError(
                    "No se pudo generar el plan de analisis. "
                    "Por favor, intenta reformular tu pregunta."
                ) from e

            # Step 3: Execute all queries
            logger.info("Step 3: Executing queries...")
            query_results = []
            for query_plan in llm_response.queries:
                result = await self._execute_single_query(query_plan)
                query_results.append(result)

            # Calculate statistics
            successful_queries = sum(1 for r in query_results if r["error"] is None)
            total_rows = sum(r["row_count"] for r in query_results if r["error"] is None)

            logger.info(
                f"Query execution complete: {successful_queries}/{len(query_results)} successful, "
                f"{total_rows} total rows"
            )

            # Check if all queries failed
            if successful_queries == 0:
                logger.error("All queries failed")
                raise AnalysisError(
                    "No se pudo ejecutar ninguna consulta exitosamente. "
                    "Por favor, intenta reformular tu pregunta."
                )

            # Step 4: Generate unified analysis
            logger.info("Step 4: Generating unified analysis...")
            try:
                analysis = await self.llm_service.generate_unified_analysis(
                    user_prompt=user_prompt,
                    query_results=query_results
                )
                logger.info("Unified analysis generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate unified analysis: {e}")
                # Fallback to a simple analysis
                analysis = f"Se ejecutaron {successful_queries} consultas exitosamente con un total de {total_rows} registros."

            # Step 5: Calculate execution time and return
            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Analysis complete: {len(query_results)} queries, "
                f"{successful_queries} successful, {execution_time_ms}ms"
            )

            return {
                "analysis": analysis,
                "queries": query_results,
                "metadata": {
                    "total_queries": len(query_results),
                    "successful_queries": successful_queries,
                    "total_rows": total_rows,
                    "execution_time_ms": execution_time_ms
                }
            }

        except AnalysisError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during analysis: {e}")
            raise AnalysisError(
                "Ocurrio un error inesperado durante el analisis. "
                "Por favor, intenta nuevamente."
            ) from e

    async def _execute_single_query(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """
        Execute a single query with validation and error handling.

        Args:
            query_plan: The query plan to execute

        Returns:
            Dictionary with query result or error
        """
        result = {
            "query_id": query_plan.query_id,
            "purpose": query_plan.purpose,
            "sql_query": query_plan.sql_query,
            "data": [],
            "row_count": 0,
            "error": None
        }

        try:
            # Validate query
            logger.info(f"Validating query {query_plan.query_id}...")
            self.query_validator.validate(query_plan.sql_query)
            sanitized_query = self.query_validator.sanitize_query(query_plan.sql_query)
            result["sql_query"] = sanitized_query

            # Execute query
            logger.info(f"Executing query {query_plan.query_id}: {sanitized_query[:80]}...")
            query_data = self.db_repository.execute_query(
                sanitized_query,
                validate=False  # Already validated
            )

            result["data"] = query_data
            result["row_count"] = len(query_data)
            logger.info(f"Query {query_plan.query_id} returned {len(query_data)} rows")

        except QueryValidationError as e:
            logger.error(f"Query {query_plan.query_id} validation failed: {e}")
            result["error"] = f"Consulta no segura: {str(e)}"

        except DatabaseConnectionError as e:
            logger.error(f"Query {query_plan.query_id} execution failed: {e}")
            result["error"] = f"Error de base de datos: {str(e)}"

        except Exception as e:
            logger.error(f"Query {query_plan.query_id} failed unexpectedly: {e}")
            result["error"] = f"Error inesperado: {str(e)}"

        return result

    async def validate_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """
        Validate a user prompt without executing full analysis.
        Useful for quick checks.

        Args:
            user_prompt: User's question

        Returns:
            Dictionary with validation results
        """
        if not user_prompt or len(user_prompt.strip()) < 5:
            return {
                "valid": False,
                "error": "El prompt debe tener al menos 5 caracteres"
            }

        if len(user_prompt) > 500:
            return {
                "valid": False,
                "error": "El prompt no debe exceder 500 caracteres"
            }

        return {
            "valid": True,
            "message": "Prompt valido"
        }

    def test_services(self) -> Dict[str, bool]:
        """
        Test connectivity of all dependent services.
        Useful for health checks.

        Returns:
            Dictionary with service status
        """
        results = {}

        # Test database connection
        try:
            results["database"] = self.db_repository.test_connection()
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            results["database"] = False

        # Test LLM service
        try:
            results["llm"] = self.llm_service.test_connection()
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            results["llm"] = False

        return results

    def get_available_tables(self) -> List[str]:
        """
        Get list of available tables in the database.

        Returns:
            List of table names
        """
        try:
            return self.db_repository.get_table_names()
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get general database information.

        Returns:
            Dictionary with database metadata
        """
        try:
            tables = self.get_available_tables()
            return {
                "table_count": len(tables),
                "tables": tables,
                "max_query_rows": self.db_repository.max_rows
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                "error": str(e)
            }
