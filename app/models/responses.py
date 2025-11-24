"""
API response models.
Defines schemas for outgoing API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChartResponse(BaseModel):
    """
    Response model for a single chart.
    """

    type: str = Field(
        ...,
        description="Chart type (bar, line, pie, scatter, heatmap)"
    )

    title: str = Field(
        ...,
        description="Chart title in Spanish"
    )

    image_base64: str = Field(
        ...,
        description="Base64-encoded PNG image"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "bar",
                    "title": "Top 10 Productos Más Vendidos",
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
                }
            ]
        }
    }


class AnalysisResponse(BaseModel):
    """
    Response model for data analysis endpoint.
    """

    explanation: str = Field(
        ...,
        description="Analysis explanation in Spanish"
    )

    sql_query: str = Field(
        ...,
        description="SQL query that was executed"
    )

    charts: List[ChartResponse] = Field(
        ...,
        description="List of generated charts"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "explanation": (
                        "El producto más vendido de la semana es Widget A con 125 unidades, "
                        "seguido por Gadget B con 98 unidades y Device C con 87 unidades."
                    ),
                    "sql_query": (
                        "SELECT TOP 10 product_name, SUM(quantity) as total_sales "
                        "FROM Sales JOIN Products ON Sales.product_id = Products.product_id "
                        "WHERE sale_date >= DATEADD(day, -7, GETDATE()) "
                        "GROUP BY product_name ORDER BY total_sales DESC"
                    ),
                    "charts": [
                        {
                            "type": "bar",
                            "title": "Top 10 Productos de la Semana",
                            "image_base64": "iVBORw0KGgo..."
                        }
                    ]
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
