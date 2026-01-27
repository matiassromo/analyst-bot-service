"""
API response models.
Defines schemas for outgoing API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryResult(BaseModel):
    """
    Result of a single query execution.
    """

    query_id: str = Field(
        ...,
        description="Unique identifier for this query"
    )

    purpose: str = Field(
        ...,
        description="Description of what this query answers"
    )

    sql_query: str = Field(
        ...,
        description="The SQL query that was executed"
    )

    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Query results as list of row dictionaries"
    )

    row_count: int = Field(
        ...,
        description="Number of rows returned"
    )

    error: Optional[str] = Field(
        None,
        description="Error message if query failed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "q1",
                    "purpose": "Obtener los productos mas vendidos",
                    "sql_query": "SELECT TOP 10 Name, TotalSales FROM Products ORDER BY TotalSales DESC",
                    "data": [{"Name": "Widget A", "TotalSales": 125}],
                    "row_count": 10,
                    "error": None
                }
            ]
        }
    }


class AnalysisMetadata(BaseModel):
    """
    Metadata about the analysis execution.
    """

    total_queries: int = Field(
        ...,
        description="Total number of queries planned"
    )

    successful_queries: int = Field(
        ...,
        description="Number of queries executed successfully"
    )

    total_rows: int = Field(
        ...,
        description="Total rows returned across all queries"
    )

    execution_time_ms: int = Field(
        ...,
        description="Total execution time in milliseconds"
    )


class MultiQueryAnalysisResponse(BaseModel):
    """
    Response model for multi-query data analysis endpoint.
    """

    analysis: str = Field(
        ...,
        description="Unified analysis text in Spanish synthesizing all query results"
    )

    queries: List[QueryResult] = Field(
        ...,
        description="List of executed queries with their results"
    )

    metadata: AnalysisMetadata = Field(
        ...,
        description="Execution metadata"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "analysis": (
                        "El producto mas vendido es Widget A con 125 unidades. "
                        "Comparado con el mes anterior, las ventas aumentaron un 15%."
                    ),
                    "queries": [
                        {
                            "query_id": "q1",
                            "purpose": "Obtener los productos mas vendidos",
                            "sql_query": "SELECT TOP 10 ...",
                            "data": [{"Name": "Widget A", "TotalSales": 125}],
                            "row_count": 10,
                            "error": None
                        },
                        {
                            "query_id": "q2",
                            "purpose": "Comparar con mes anterior",
                            "sql_query": "SELECT SUM(...)",
                            "data": [{"LastMonthSales": 850}],
                            "row_count": 1,
                            "error": None
                        }
                    ],
                    "metadata": {
                        "total_queries": 2,
                        "successful_queries": 2,
                        "total_rows": 11,
                        "execution_time_ms": 245
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Response model for errors.
    """

    error: str = Field(
        ...,
        description="Error message"
    )

    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

    type: Optional[str] = Field(
        None,
        description="Error type classification"
    )


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """

    status: str = Field(
        ...,
        description="Overall service status (healthy, degraded, unhealthy)"
    )

    service: str = Field(
        ...,
        description="Service name"
    )

    version: str = Field(
        ...,
        description="Service version"
    )

    environment: str = Field(
        ...,
        description="Environment (development, production)"
    )

    services: Optional[Dict[str, bool]] = Field(
        None,
        description="Status of dependent services"
    )


class DatabaseInfoResponse(BaseModel):
    """
    Response model for database information.
    """

    table_count: int = Field(
        ...,
        description="Number of tables in the database"
    )

    tables: List[str] = Field(
        ...,
        description="List of table names"
    )

    max_query_rows: int = Field(
        ...,
        description="Maximum rows that can be returned by a query"
    )


class ValidationResponse(BaseModel):
    """
    Response model for validation checks.
    """

    valid: bool = Field(
        ...,
        description="Whether the validation passed"
    )

    message: Optional[str] = Field(
        None,
        description="Validation message"
    )

    error: Optional[str] = Field(
        None,
        description="Validation error if validation failed"
    )
