"""
Pytest configuration and fixtures.
Provides test fixtures for mocking services and dependencies.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from app.main import app
from app.models.llm_models import LLMMultiQueryResponse, QueryPlan


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
- ProductId (int, PRIMARY KEY)
- Name (nvarchar(100), NOT NULL)
- Price (decimal)

Table: Sales
- SaleId (int, PRIMARY KEY)
- ProductId (int, FOREIGN KEY -> Products.ProductId)
- Quantity (int)
- SaleDate (datetime)
"""

    # Mock query results
    mock_repo.execute_query.return_value = [
        {"Name": "Widget A", "TotalSales": 125},
        {"Name": "Gadget B", "TotalSales": 98},
        {"Name": "Device C", "TotalSales": 87}
    ]

    mock_repo.test_connection.return_value = True
    mock_repo.get_table_names.return_value = ["Products", "Sales"]
    mock_repo.max_rows = 1000

    return mock_repo


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service for testing multi-query functionality.
    """
    mock_service = Mock()

    # Create mock multi-query response
    mock_response = LLMMultiQueryResponse(
        queries=[
            QueryPlan(
                query_id="q1",
                purpose="Obtener los productos mas vendidos",
                sql_query=(
                    "SELECT TOP 10 p.Name, SUM(s.Quantity) as TotalSales "
                    "FROM Sales s JOIN Products p ON s.ProductId = p.ProductId "
                    "GROUP BY p.Name ORDER BY TotalSales DESC"
                )
            ),
            QueryPlan(
                query_id="q2",
                purpose="Comparar con mes anterior",
                sql_query=(
                    "SELECT SUM(Quantity) as LastMonthSales FROM Sales "
                    "WHERE SaleDate >= DATEADD(month, -1, GETDATE())"
                )
            )
        ]
    )

    # Mock async methods
    mock_service.generate_multi_query_plan = AsyncMock(return_value=mock_response)
    mock_service.generate_unified_analysis = AsyncMock(
        return_value=(
            "Los productos mas vendidos son Widget A con 125 unidades, "
            "Gadget B con 98 unidades y Device C con 87 unidades. "
            "En comparacion con el mes anterior, las ventas se mantienen estables."
        )
    )
    mock_service.test_connection.return_value = True

    return mock_service


@pytest.fixture
def mock_llm_service_single_query():
    """
    Mock LLM service that returns a single query.
    """
    mock_service = Mock()

    mock_response = LLMMultiQueryResponse(
        queries=[
            QueryPlan(
                query_id="q1",
                purpose="Obtener el total de ventas",
                sql_query="SELECT SUM(Quantity) as TotalSales FROM Sales"
            )
        ]
    )

    mock_service.generate_multi_query_plan = AsyncMock(return_value=mock_response)
    mock_service.generate_unified_analysis = AsyncMock(
        return_value="El total de ventas es de 310 unidades."
    )
    mock_service.test_connection.return_value = True

    return mock_service


@pytest.fixture
def mock_query_validator():
    """
    Mock query validator for testing.
    """
    mock_validator = Mock()

    mock_validator.validate.return_value = None
    mock_validator.sanitize_query.side_effect = lambda q: q  # Return query as-is

    return mock_validator


@pytest.fixture
def sample_query_results() -> List[Dict[str, Any]]:
    """
    Sample query results for testing.
    """
    return [
        {"Name": "Widget A", "TotalSales": 125},
        {"Name": "Gadget B", "TotalSales": 98},
        {"Name": "Device C", "TotalSales": 87},
        {"Name": "Tool D", "TotalSales": 76},
        {"Name": "Part E", "TotalSales": 65}
    ]


@pytest.fixture
def sample_multi_query_response() -> Dict[str, Any]:
    """
    Sample multi-query analysis response for testing.
    """
    return {
        "analysis": (
            "Los productos mas vendidos son Widget A con 125 unidades, "
            "seguido por Gadget B con 98 unidades."
        ),
        "queries": [
            {
                "query_id": "q1",
                "purpose": "Obtener los productos mas vendidos",
                "sql_query": "SELECT TOP 10 ...",
                "data": [
                    {"Name": "Widget A", "TotalSales": 125},
                    {"Name": "Gadget B", "TotalSales": 98}
                ],
                "row_count": 2,
                "error": None
            },
            {
                "query_id": "q2",
                "purpose": "Obtener ventas del mes anterior",
                "sql_query": "SELECT SUM(...)",
                "data": [{"LastMonthSales": 850}],
                "row_count": 1,
                "error": None
            }
        ],
        "metadata": {
            "total_queries": 2,
            "successful_queries": 2,
            "total_rows": 3,
            "execution_time_ms": 150
        }
    }


@pytest.fixture
def sample_partial_failure_response() -> Dict[str, Any]:
    """
    Sample response where one query failed.
    """
    return {
        "analysis": "Se obtuvo informacion parcial debido a un error en una consulta.",
        "queries": [
            {
                "query_id": "q1",
                "purpose": "Obtener los productos mas vendidos",
                "sql_query": "SELECT TOP 10 ...",
                "data": [{"Name": "Widget A", "TotalSales": 125}],
                "row_count": 1,
                "error": None
            },
            {
                "query_id": "q2",
                "purpose": "Consulta con error",
                "sql_query": "SELECT * FROM NonExistentTable",
                "data": [],
                "row_count": 0,
                "error": "Error de base de datos: Table not found"
            }
        ],
        "metadata": {
            "total_queries": 2,
            "successful_queries": 1,
            "total_rows": 1,
            "execution_time_ms": 100
        }
    }
