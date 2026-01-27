"""
API request models.
Defines schemas for incoming API requests.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class AnalysisRequest(BaseModel):
    """
    Request model for data analysis endpoint.
    """

    prompt: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Business question in Spanish about the data"
    )

    exclude_tables: Optional[List[str]] = Field(
        None,
        description="Optional list of table names to exclude from analysis"
    )

    max_queries: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum number of queries to generate (1-5)"
    )

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and clean the prompt."""
        v = v.strip()

        if len(v) < 5:
            raise ValueError("El prompt debe tener al menos 5 caracteres")

        if len(v) > 500:
            raise ValueError("El prompt no debe exceder 500 caracteres")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Cual es el producto mas vendido de la semana?",
                    "exclude_tables": None,
                    "max_queries": 5
                },
                {
                    "prompt": "Cuales son las ventas totales por categoria este mes y como se comparan con el mes pasado?",
                    "exclude_tables": ["SensitiveData", "InternalLogs"],
                    "max_queries": 3
                }
            ]
        }
    }


class HealthCheckRequest(BaseModel):
    """
    Request model for detailed health check.
    """

    include_services: bool = Field(
        default=False,
        description="Include status of dependent services (database, LLM)"
    )
