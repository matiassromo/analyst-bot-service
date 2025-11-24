"""
Pydantic models for LLM request and response structures.
Defines structured outputs for Gemini API responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


class ChartConfig(BaseModel):
    """
    Configuration for a single chart/visualization.

    Defines chart type and display parameters.
    Used by ChartService to generate visualizations.
    """

    type: Literal["bar", "line", "pie", "scatter", "heatmap"] = Field(
        ...,
        description="Type of chart to generate"
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Chart title in Spanish"
    )

    x_column: str = Field(
        ...,
        description="Column name for X-axis data"
    )

    y_column: Optional[str] = Field(
        None,
        description="Column name for Y-axis data (not needed for pie charts)"
    )

    x_label: Optional[str] = Field(
        None,
        description="Label for X-axis (defaults to x_column if not provided)"
    )

    y_label: Optional[str] = Field(
        None,
        description="Label for Y-axis (defaults to y_column if not provided)"
    )

    color_palette: str = Field(
        default="viridis",
        description="Color palette for the chart (viridis, Set2, husl, etc.)"
    )

    @field_validator('type')
    @classmethod
    def validate_chart_type(cls, v: str) -> str:
        """Ensure chart type is supported."""
        allowed_types = ["bar", "line", "pie", "scatter", "heatmap"]
        if v not in allowed_types:
            raise ValueError(
                f"Chart type must be one of {allowed_types}, got: {v}"
            )
        return v

    @field_validator('y_column')
    @classmethod
    def validate_y_column(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure y_column is provided for charts that need it."""
        chart_type = info.data.get('type')
        if chart_type != 'pie' and not v:
            raise ValueError(
                f"y_column is required for {chart_type} charts"
            )
        return v


class LLMAnalysisResponse(BaseModel):
    """
    Structured response from Gemini LLM for data analysis.

    Contains:
    - SQL query to execute
    - Human-readable explanation in Spanish
    - Chart configurations for visualization
    """

    sql_query: str = Field(
        ...,
        min_length=10,
        description="SQL Server query to answer the user's question"
    )

    explanation: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Clear explanation of the analysis in Spanish"
    )

    chart_configs: List[ChartConfig] = Field(
        ...,
        min_length=0,
        max_length=3,
        description="List of chart configurations (max 3 charts)"
    )

    @field_validator('sql_query')
    @classmethod
    def validate_sql_query(cls, v: str) -> str:
        """Basic SQL query validation."""
        v = v.strip()
        if not v.upper().startswith('SELECT'):
            raise ValueError(
                "SQL query must start with SELECT"
            )
        return v

    @field_validator('chart_configs')
    @classmethod
    def validate_chart_count(cls, v: List[ChartConfig]) -> List[ChartConfig]:
        """Ensure we don't have too many charts."""
        if len(v) > 3:
            raise ValueError(
                f"Maximum 3 charts allowed, got {len(v)}"
            )
        return v


class LLMPromptContext(BaseModel):
    """
    Context information for LLM prompts.
    """

    database_schema: str = Field(
        ...,
        description="Formatted database schema"
    )

    user_prompt: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="User's question in Spanish"
    )

    additional_context: Optional[str] = Field(
        None,
        description="Additional context or constraints"
    )


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMAPIError(LLMError):
    """Raised when Gemini API call fails."""
    pass


class LLMParseError(LLMError):
    """Raised when LLM response cannot be parsed."""
    pass


class LLMValidationError(LLMError):
    """Raised when LLM response fails validation."""
    pass
