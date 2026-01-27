"""
Analysis API routes.
Endpoints for multi-query data analysis, health checks, and database info.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.requests import AnalysisRequest, HealthCheckRequest
from app.models.responses import (
    MultiQueryAnalysisResponse,
    QueryResult,
    AnalysisMetadata,
    HealthCheckResponse,
    DatabaseInfoResponse,
    ErrorResponse,
    ValidationResponse
)
from app.services.analysis_service import AnalysisService, AnalysisError
from app.api.dependencies import get_analysis_service
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analysis",
    response_model=MultiQueryAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or query validation failed"},
        500: {"model": ErrorResponse, "description": "Analysis failed"},
        502: {"model": ErrorResponse, "description": "LLM API error"},
        503: {"model": ErrorResponse, "description": "Database connection error"}
    },
    summary="Analyze business data with multiple queries",
    description=(
        "Analyze business data using natural language prompts. "
        "Generates 1-5 SQL queries to comprehensively answer the question, "
        "executes them, and returns a unified analysis."
    )
)
async def analyze_data(
    request: AnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> MultiQueryAnalysisResponse:
    """
    Analyze business data using a natural language prompt.

    **Process:**
    1. Loads database schema
    2. Uses LLM to generate 1-5 SQL queries
    3. Validates and executes each query
    4. Generates unified analysis from all results
    5. Returns analysis text and all query results

    **Example Request:**
    ```json
    {
      "prompt": "Cuales son los productos mas vendidos y como se comparan con el mes pasado?",
      "exclude_tables": null,
      "max_queries": 5
    }
    ```

    **Parameters:**
    - `prompt`: Your question in Spanish
    - `exclude_tables`: Optional list of tables to exclude from analysis
    - `max_queries`: Maximum queries to generate (1-5, default 5)

    **Example Response:**
    ```json
    {
      "analysis": "El producto mas vendido es Widget A con 125 unidades...",
      "queries": [
        {
          "query_id": "q1",
          "purpose": "Obtener los productos mas vendidos",
          "sql_query": "SELECT TOP 10 ...",
          "data": [{"Name": "Widget A", "TotalSales": 125}],
          "row_count": 10,
          "error": null
        }
      ],
      "metadata": {
        "total_queries": 2,
        "successful_queries": 2,
        "total_rows": 11,
        "execution_time_ms": 245
      }
    }
    ```
    """
    try:
        logger.info(f"Analysis request received: {request.prompt[:100]}...")

        # Perform analysis
        result = await analysis_service.analyze(
            user_prompt=request.prompt,
            exclude_tables=request.exclude_tables,
            max_queries=request.max_queries
        )

        logger.info("Analysis completed successfully")

        # Build response
        return MultiQueryAnalysisResponse(
            analysis=result["analysis"],
            queries=[QueryResult(**q) for q in result["queries"]],
            metadata=AnalysisMetadata(**result["metadata"])
        )

    except AnalysisError as e:
        logger.error(f"Analysis error: {e}")

        # Determine appropriate status code based on error
        error_msg = str(e)

        if "base de datos" in error_msg.lower() or "database" in error_msg.lower():
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif "consulta" in error_msg.lower() or "query" in error_msg.lower():
            status_code = status.HTTP_400_BAD_REQUEST
        elif "llm" in error_msg.lower() or "generar" in error_msg.lower():
            status_code = status.HTTP_502_BAD_GATEWAY
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        raise HTTPException(
            status_code=status_code,
            detail=str(e)
        )

    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor. Por favor, intenta nuevamente."
        )


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate a prompt",
    description="Validate a prompt without executing full analysis"
)
async def validate_prompt(
    request: AnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> ValidationResponse:
    """
    Validate a user prompt without executing the full analysis.
    Useful for checking prompt validity before submission.
    """
    try:
        result = await analysis_service.validate_prompt(request.prompt)
        return ValidationResponse(**result)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ValidationResponse(
            valid=False,
            error=str(e)
        )


@router.get(
    "/database/info",
    response_model=DatabaseInfoResponse,
    summary="Get database information",
    description="Get information about available database tables"
)
async def get_database_info(
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> DatabaseInfoResponse:
    """
    Get information about the connected database.

    Returns:
    - Number of tables
    - List of table names
    - Maximum query row limit
    """
    try:
        info = analysis_service.get_database_info()

        if "error" in info:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No se pudo obtener informacion de la base de datos: {info['error']}"
            )

        return DatabaseInfoResponse(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener informacion de la base de datos"
        )


@router.post(
    "/health/detailed",
    response_model=HealthCheckResponse,
    summary="Detailed health check",
    description="Check health of the service and its dependencies"
)
async def detailed_health_check(
    request: HealthCheckRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> HealthCheckResponse:
    """
    Perform a detailed health check of the service.

    Optionally tests connectivity to:
    - Database (SQL Server)
    - LLM API (Gemini)
    """
    response_data = {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }

    if request.include_services:
        try:
            service_status = analysis_service.test_services()
            response_data["services"] = service_status

            # Determine overall status
            if all(service_status.values()):
                response_data["status"] = "healthy"
            elif any(service_status.values()):
                response_data["status"] = "degraded"
            else:
                response_data["status"] = "unhealthy"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            response_data["status"] = "unhealthy"
            response_data["services"] = {
                "database": False,
                "llm": False
            }

    return HealthCheckResponse(**response_data)


@router.get(
    "/",
    summary="API information",
    description="Get information about available endpoints"
)
async def api_info():
    """
    Get information about the Analysis API.
    """
    return {
        "name": "Analysis API",
        "version": settings.app_version,
        "endpoints": {
            "POST /analysis": "Analyze data with natural language prompt (multi-query)",
            "POST /validate": "Validate a prompt without executing",
            "GET /database/info": "Get database information",
            "POST /health/detailed": "Detailed health check",
            "GET /": "This endpoint"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }
