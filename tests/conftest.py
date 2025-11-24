"""
Pytest configuration and fixtures.
Provides test fixtures for mocking services and dependencies.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from app.main import app
from app.models.llm_models import LLMAnalysisResponse, ChartConfig


@pytest.fixture
def test_client():
    """
    FastAPI test client fixture.
    """
    return TestClient(app)


@pytest.fixture
def mock_database_repository():
    """
    Mock database repository for testing.
    """
    mock_repo = Mock()

    # Mock schema
    mock_repo.get_schema.return_value = """
Database Schema:

Table: Products
- product_id (int, PRIMARY KEY)
- product_name (nvarchar(100), NOT NULL)
- price (decimal)

Table: Sales
- sale_id (int, PRIMARY KEY)
- product_id (int, FOREIGN KEY -> Products.product_id)
- quantity (int)
- sale_date (datetime)
"""

    # Mock query results
    mock_repo.execute_query.return_value = [
        {"product_name": "Widget A", "total_sales": 125},
        {"product_name": "Gadget B", "total_sales": 98},
        {"product_name": "Device C", "total_sales": 87}
    ]

    mock_repo.test_connection.return_value = True
    mock_repo.get_table_names.return_value = ["Products", "Sales"]

    return mock_repo


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service for testing.
    """
    mock_service = Mock()

    # Create mock LLM response
    mock_response = LLMAnalysisResponse(
        sql_query=(
            "SELECT TOP 10 product_name, SUM(quantity) as total_sales "
            "FROM Sales JOIN Products ON Sales.product_id = Products.product_id "
            "GROUP BY product_name ORDER BY total_sales DESC"
        ),
        explanation=(
            "Los productos más vendidos son Widget A con 125 unidades, "
            "Gadget B con 98 unidades y Device C con 87 unidades."
        ),
        chart_configs=[
            ChartConfig(
                type="bar",
                title="Top 10 Productos Más Vendidos",
                x_column="product_name",
                y_column="total_sales",
                x_label="Producto",
                y_label="Ventas Totales",
                color_palette="viridis"
            )
        ]
    )

    # Mock async method
    mock_service.generate_analysis = AsyncMock(return_value=mock_response)
    mock_service.test_connection.return_value = True

    return mock_service


@pytest.fixture
def mock_chart_service():
    """
    Mock chart service for testing.
    """
    mock_service = Mock()

    # Return fake base64 image
    mock_service.generate_chart.return_value = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    return mock_service


@pytest.fixture
def mock_query_validator():
    """
    Mock query validator for testing.
    """
    mock_validator = Mock()

    mock_validator.validate.return_value = None
    mock_validator.sanitize_query.return_value = (
        "SELECT TOP 10 product_name, SUM(quantity) as total_sales "
        "FROM Sales JOIN Products ON Sales.product_id = Products.product_id "
        "GROUP BY product_name ORDER BY total_sales DESC"
    )

    return mock_validator


@pytest.fixture
def sample_query_results() -> List[Dict[str, Any]]:
    """
    Sample query results for testing chart generation.
    """
    return [
        {"product_name": "Widget A", "total_sales": 125},
        {"product_name": "Gadget B", "total_sales": 98},
        {"product_name": "Device C", "total_sales": 87},
        {"product_name": "Tool D", "total_sales": 76},
        {"product_name": "Part E", "total_sales": 65}
    ]


@pytest.fixture
def sample_chart_config() -> ChartConfig:
    """
    Sample chart configuration for testing.
    """
    return ChartConfig(
        type="bar",
        title="Test Bar Chart",
        x_column="product_name",
        y_column="total_sales",
        x_label="Product",
        y_label="Sales",
        color_palette="viridis"
    )
