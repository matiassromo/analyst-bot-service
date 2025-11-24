"""
Analysis service - Main orchestrator for the analysis workflow.
Coordinates LLM, database, and chart generation services.
"""

import logging
from typing import List, Dict, Any

from app.services.llm_service import LLMService, LLMAPIError, LLMParseError
from app.services.chart_service import ChartService, ChartGenerationError
from app.repositories.database_repository import (
    DatabaseRepository,
    DatabaseConnectionError
)
from app.utils.query_validator import QueryValidator, QueryValidationError

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base exception for analysis errors."""
    pass


class AnalysisService:
    """
    Main service orchestrator for data analysis workflow.

    Workflow:
    1. Get database schema
    2. Call LLM to generate SQL query and chart configs
    3. Validate SQL query
    4. Execute query against database
    5. Generate charts from results
    6. Return structured response
    """

    def __init__(
        self,
        llm_service: LLMService,
        chart_service: ChartService,
        db_repository: DatabaseRepository,
        query_validator: QueryValidator = None
    ):
        """
        Initialize analysis service with dependencies.

        Args:
            llm_service: LLM service for query generation
            chart_service: Chart service for visualization
            db_repository: Database repository for query execution
            query_validator: Query validator (optional, creates new if not provided)
        """
        self.llm_service = llm_service
        self.chart_service = chart_service
        self.db_repository = db_repository
        self.query_validator = query_validator or QueryValidator()

        logger.info("Analysis service initialized")

    async def analyze(
        self,
        user_prompt: str,
        exclude_tables: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform complete analysis workflow.

        Args:
            user_prompt: User's question in Spanish
            exclude_tables: Tables to exclude from schema (optional)

        Returns:
            Dictionary with:
            - explanation: Analysis explanation in Spanish
            - sql_query: Executed SQL query
            - charts: List of generated charts with base64 images

        Raises:
            AnalysisError: If any step in the workflow fails
        """
        try:
            logger.info(f"Starting analysis for prompt: {user_prompt[:100]}...")

            # Step 1: Get database schema
            logger.info("Step 1: Loading database schema...")
            try:
                schema = self.db_repository.get_schema(exclude_tables)
                logger.debug(f"Schema loaded: {len(schema)} characters")
            except DatabaseConnectionError as e:
                logger.error(f"Failed to load schema: {e}")
                raise AnalysisError(
                    "No se pudo conectar a la base de datos. "
                    "Por favor, verifica la configuración."
                ) from e

            # Step 2: Call LLM for analysis
            logger.info("Step 2: Generating analysis with LLM...")
            try:
                llm_response = await self.llm_service.generate_analysis(
                    user_prompt=user_prompt,
                    database_schema=schema
                )
                logger.info(f"LLM generated query: {llm_response.sql_query[:100]}...")
                logger.info(f"LLM generated {len(llm_response.chart_configs)} chart configs")
            except (LLMAPIError, LLMParseError) as e:
                logger.error(f"LLM generation failed: {e}")
                raise AnalysisError(
                    "No se pudo generar el análisis. "
                    "Por favor, intenta reformular tu pregunta."
                ) from e

            # Step 3: Validate SQL query
            logger.info("Step 3: Validating SQL query...")
            try:
                self.query_validator.validate(llm_response.sql_query)
                sanitized_query = self.query_validator.sanitize_query(
                    llm_response.sql_query
                )
                logger.info("Query validation successful")
            except QueryValidationError as e:
                logger.error(f"Query validation failed: {e}")
                raise AnalysisError(
                    f"La consulta generada no es segura: {str(e)}"
                ) from e

            # Step 4: Execute query
            logger.info("Step 4: Executing query against database...")
            try:
                query_results = self.db_repository.execute_query(
                    sanitized_query,
                    validate=False  # Already validated
                )
                logger.info(f"Query executed successfully: {len(query_results)} rows returned")

                # Check if we got results
                if not query_results:
                    logger.warning("Query returned no results")
                    return {
                        "explanation": (
                            llm_response.explanation +
                            "\n\nNota: La consulta no retornó ningún resultado."
                        ),
                        "sql_query": sanitized_query,
                        "charts": []
                    }

            except DatabaseConnectionError as e:
                logger.error(f"Query execution failed: {e}")
                raise AnalysisError(
                    f"Error al ejecutar la consulta: {str(e)}"
                ) from e

            # Step 5: Generate charts
            logger.info("Step 5: Generating charts...")
            charts = []

            for i, chart_config in enumerate(llm_response.chart_configs):
                try:
                    logger.info(
                        f"Generating chart {i+1}/{len(llm_response.chart_configs)}: "
                        f"{chart_config.type} - {chart_config.title}"
                    )

                    image_base64 = self.chart_service.generate_chart(
                        data=query_results,
                        config=chart_config
                    )

                    charts.append({
                        "type": chart_config.type,
                        "title": chart_config.title,
                        "image_base64": image_base64
                    })

                    logger.info(f"Chart {i+1} generated successfully")

                except ChartGenerationError as e:
                    logger.error(
                        f"Failed to generate chart '{chart_config.title}': {e}"
                    )
                    # Continue with other charts instead of failing completely
                    continue

            if not charts:
                logger.warning("No charts were generated")

            # Step 6: Return structured response
            logger.info(
                f"Analysis complete: {len(query_results)} rows, "
                f"{len(charts)} charts generated"
            )

            return {
                "explanation": llm_response.explanation,
                "sql_query": sanitized_query,
                "charts": charts
            }

        except AnalysisError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during analysis: {e}")
            raise AnalysisError(
                "Ocurrió un error inesperado durante el análisis. "
                "Por favor, intenta nuevamente."
            ) from e

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
            "message": "Prompt válido"
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

        # Chart service is always available (no external dependency)
        results["chart"] = True

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
