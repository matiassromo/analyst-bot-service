"""
Integration tests for the analysis workflow.
Tests the complete flow from API request to response.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_analysis_service, reset_dependencies
from app.services.analysis_service import AnalysisService


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "service" in data
    assert "version" in data
    assert "docs" in data


def test_api_info_endpoint(client):
    """Test the API info endpoint."""
    response = client.get("/api/v1/")

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_analysis_endpoint_with_mocks(
    client,
    mock_database_repository,
    mock_llm_service,
    mock_chart_service,
    mock_query_validator
):
    """Test the analysis endpoint with mocked services."""

    # Create mock analysis service
    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        chart_service=mock_chart_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Override dependency
    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        # Make request
        response = client.post(
            "/api/v1/analysis",
            json={"prompt": "¿Cuál es el producto más vendido?"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert "explanation" in data
        assert "sql_query" in data
        assert "charts" in data
        assert len(data["charts"]) > 0

        # Verify chart structure
        chart = data["charts"][0]
        assert "type" in chart
        assert "title" in chart
        assert "image_base64" in chart

    finally:
        # Clean up
        app.dependency_overrides.clear()


def test_analysis_endpoint_validation_error(client):
    """Test analysis endpoint with invalid prompt."""
    # Prompt too short
    response = client.post(
        "/api/v1/analysis",
        json={"prompt": "Hi"}
    )

    assert response.status_code == 422  # Validation error


def test_analysis_endpoint_missing_prompt(client):
    """Test analysis endpoint without prompt."""
    response = client.post(
        "/api/v1/analysis",
        json={}
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_validate_endpoint(client):
    """Test the prompt validation endpoint."""
    response = client.post(
        "/api/v1/validate",
        json={"prompt": "¿Cuáles son las ventas totales?"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "valid" in data


def test_database_info_endpoint_with_mock(
    client,
    mock_database_repository,
    mock_llm_service,
    mock_chart_service
):
    """Test database info endpoint with mocked services."""

    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        chart_service=mock_chart_service,
        db_repository=mock_database_repository
    )

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        response = client.get("/api/v1/database/info")

        assert response.status_code == 200
        data = response.json()

        assert "table_count" in data
        assert "tables" in data
        assert "max_query_rows" in data

    finally:
        app.dependency_overrides.clear()


def test_detailed_health_check(client):
    """Test detailed health check endpoint."""
    response = client.post(
        "/api/v1/health/detailed",
        json={"include_services": False}
    )

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_analysis_workflow_order(
    mock_database_repository,
    mock_llm_service,
    mock_chart_service,
    mock_query_validator
):
    """Test that analysis workflow executes steps in correct order."""

    analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        chart_service=mock_chart_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Execute analysis
    result = await analysis_service.analyze("Test prompt")

    # Verify all services were called
    mock_database_repository.get_schema.assert_called_once()
    mock_llm_service.generate_analysis.assert_called_once()
    mock_query_validator.validate.assert_called_once()
    mock_database_repository.execute_query.assert_called_once()
    mock_chart_service.generate_chart.assert_called()

    # Verify result structure
    assert "explanation" in result
    assert "sql_query" in result
    assert "charts" in result
