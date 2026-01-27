"""
Integration tests for the multi-query analysis workflow.
Tests the complete flow from API request to response.
"""

import pytest
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_analysis_service, reset_dependencies
from app.services.analysis_service import AnalysisService
from app.models.llm_models import LLMMultiQueryResponse, QueryPlan


# Headers to bypass IP whitelist in tests
WHITELIST_HEADERS = {"X-Forwarded-For": "127.0.0.1"}


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
    response = client.get("/", headers=WHITELIST_HEADERS)

    assert response.status_code == 200
    data = response.json()

    assert "service" in data
    assert "version" in data
    assert "docs" in data


def test_api_info_endpoint(client):
    """Test the API info endpoint."""
    response = client.get("/api/v1/", headers=WHITELIST_HEADERS)

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_analysis_endpoint_with_mocks(
    client,
    mock_database_repository,
    mock_llm_service,
    mock_query_validator
):
    """Test the analysis endpoint with mocked services."""

    # Create mock analysis service
    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Override dependency
    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        # Make request
        response = client.post(
            "/api/v1/analysis",
            json={"prompt": "Cuales son los productos mas vendidos?"},
            headers=WHITELIST_HEADERS
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Verify new response structure
        assert "analysis" in data
        assert "queries" in data
        assert "metadata" in data

        # Verify analysis text
        assert len(data["analysis"]) > 0

        # Verify queries structure
        assert len(data["queries"]) >= 1
        query = data["queries"][0]
        assert "query_id" in query
        assert "purpose" in query
        assert "sql_query" in query
        assert "data" in query
        assert "row_count" in query
        assert "error" in query

        # Verify metadata
        assert "total_queries" in data["metadata"]
        assert "successful_queries" in data["metadata"]
        assert "total_rows" in data["metadata"]
        assert "execution_time_ms" in data["metadata"]

    finally:
        # Clean up
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analysis_endpoint_single_query(
    client,
    mock_database_repository,
    mock_llm_service_single_query,
    mock_query_validator
):
    """Test analysis endpoint with a single query response."""

    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service_single_query,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        response = client.post(
            "/api/v1/analysis",
            json={"prompt": "Cual es el total de ventas?"},
            headers=WHITELIST_HEADERS
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["queries"]) == 1
        assert data["metadata"]["total_queries"] == 1

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_analysis_endpoint_with_max_queries(
    client,
    mock_database_repository,
    mock_llm_service,
    mock_query_validator
):
    """Test analysis endpoint with max_queries parameter."""

    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        response = client.post(
            "/api/v1/analysis",
            json={
                "prompt": "Cuales son los productos mas vendidos?",
                "max_queries": 3
            },
            headers=WHITELIST_HEADERS
        )

        assert response.status_code == 200
        # Verify max_queries was passed (mock doesn't enforce limit)
        mock_llm_service.generate_multi_query_plan.assert_called_once()

    finally:
        app.dependency_overrides.clear()


def test_analysis_endpoint_validation_error(client):
    """Test analysis endpoint with invalid prompt."""
    # Prompt too short
    response = client.post(
        "/api/v1/analysis",
        json={"prompt": "Hi"},
        headers=WHITELIST_HEADERS
    )

    assert response.status_code == 422  # Validation error


def test_analysis_endpoint_missing_prompt(client):
    """Test analysis endpoint without prompt."""
    response = client.post(
        "/api/v1/analysis",
        json={},
        headers=WHITELIST_HEADERS
    )

    assert response.status_code == 422  # Validation error


def test_analysis_endpoint_invalid_max_queries(client):
    """Test analysis endpoint with invalid max_queries."""
    # max_queries too high
    response = client.post(
        "/api/v1/analysis",
        json={"prompt": "Cuales son los productos mas vendidos?", "max_queries": 10},
        headers=WHITELIST_HEADERS
    )

    assert response.status_code == 422  # Validation error

    # max_queries too low
    response = client.post(
        "/api/v1/analysis",
        json={"prompt": "Cuales son los productos mas vendidos?", "max_queries": 0},
        headers=WHITELIST_HEADERS
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_validate_endpoint(client):
    """Test the prompt validation endpoint."""
    response = client.post(
        "/api/v1/validate",
        json={"prompt": "Cuales son las ventas totales?"},
        headers=WHITELIST_HEADERS
    )

    assert response.status_code == 200
    data = response.json()

    assert "valid" in data


def test_database_info_endpoint_with_mock(
    client,
    mock_database_repository,
    mock_llm_service
):
    """Test database info endpoint with mocked services."""

    mock_analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository
    )

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis_service

    try:
        response = client.get("/api/v1/database/info", headers=WHITELIST_HEADERS)

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
        json={"include_services": False},
        headers=WHITELIST_HEADERS
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
    mock_query_validator
):
    """Test that analysis workflow executes steps in correct order."""

    analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Execute analysis
    result = await analysis_service.analyze("Test prompt")

    # Verify all services were called
    mock_database_repository.get_schema.assert_called_once()
    mock_llm_service.generate_multi_query_plan.assert_called_once()
    mock_query_validator.validate.assert_called()
    mock_database_repository.execute_query.assert_called()
    mock_llm_service.generate_unified_analysis.assert_called_once()

    # Verify result structure
    assert "analysis" in result
    assert "queries" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_analysis_partial_failure(
    mock_database_repository,
    mock_llm_service,
    mock_query_validator
):
    """Test analysis with partial query failure (some succeed, some fail)."""

    # Make the second query execution fail
    call_count = [0]

    def mock_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [{"Name": "Widget A", "TotalSales": 125}]
        else:
            raise Exception("Database error")

    mock_database_repository.execute_query.side_effect = mock_execute

    analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Execute analysis - should succeed with partial results
    result = await analysis_service.analyze("Test prompt")

    # Verify partial success
    assert result["metadata"]["successful_queries"] == 1
    assert result["metadata"]["total_queries"] == 2

    # Verify one query has error
    errors = [q for q in result["queries"] if q["error"] is not None]
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_analysis_all_queries_fail(
    mock_database_repository,
    mock_llm_service,
    mock_query_validator
):
    """Test analysis when all queries fail raises error."""
    from app.services.analysis_service import AnalysisError

    # Make all query executions fail
    mock_database_repository.execute_query.side_effect = Exception("Database error")

    analysis_service = AnalysisService(
        llm_service=mock_llm_service,
        db_repository=mock_database_repository,
        query_validator=mock_query_validator
    )

    # Execute analysis - should raise error
    with pytest.raises(AnalysisError) as exc_info:
        await analysis_service.analyze("Test prompt")

    assert "ninguna consulta" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_multi_query_response_structure(sample_multi_query_response):
    """Test multi-query response structure validation."""
    from app.models.responses import (
        MultiQueryAnalysisResponse,
        QueryResult,
        AnalysisMetadata
    )

    # Build response from sample data
    response = MultiQueryAnalysisResponse(
        analysis=sample_multi_query_response["analysis"],
        queries=[QueryResult(**q) for q in sample_multi_query_response["queries"]],
        metadata=AnalysisMetadata(**sample_multi_query_response["metadata"])
    )

    # Verify structure
    assert response.analysis == sample_multi_query_response["analysis"]
    assert len(response.queries) == 2
    assert response.metadata.total_queries == 2
    assert response.metadata.successful_queries == 2


@pytest.mark.asyncio
async def test_query_result_with_error():
    """Test QueryResult model with error field."""
    from app.models.responses import QueryResult

    result = QueryResult(
        query_id="q1",
        purpose="Test query",
        sql_query="SELECT 1",
        data=[],
        row_count=0,
        error="Database connection failed"
    )

    assert result.error == "Database connection failed"
    assert result.row_count == 0
    assert result.data == []
