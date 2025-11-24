"""
LLM service for Google Gemini API integration.
Generates SQL queries and analysis from natural language prompts.
"""

import json
import logging
from typing import Optional
import google.generativeai as genai

from app.core.config import settings
from app.models.llm_models import (
    LLMAnalysisResponse,
    LLMPromptContext,
    LLMAPIError,
    LLMParseError,
    LLMValidationError
)

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Google Gemini LLM.
    Generates structured analysis responses from natural language prompts.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an expert SQL analyst for a business database. Given a database schema and a user question in Spanish, you must:

1. Write a SQL Server query to answer the question
2. Provide a clear explanation in Spanish
3. Suggest appropriate chart visualizations

Respond ONLY with valid JSON matching this exact structure:
{{
  "sql_query": "SELECT TOP 10 product_name, SUM(quantity) as total_sales FROM Sales JOIN Products ON Sales.product_id = Products.product_id GROUP BY product_name ORDER BY total_sales DESC",
  "explanation": "Los productos más vendidos de la semana son Widget A con 125 unidades, seguido por Gadget B con 98 unidades.",
  "chart_configs": [
    {{
      "type": "bar",
      "title": "Top 10 Productos Más Vendidos",
      "x_column": "product_name",
      "y_column": "total_sales",
      "x_label": "Producto",
      "y_label": "Ventas Totales",
      "color_palette": "viridis"
    }}
  ]
}}

CRITICAL RULES:
- Use SQL Server syntax (TOP instead of LIMIT, GETDATE() instead of NOW())
- Always include column aliases for aggregates (e.g., SUM(quantity) as total_sales)
- Explanation must be in Spanish, clear and concise
- Maximum 3 charts per analysis
- Choose chart types appropriate for the data:
  * bar: categorical comparisons (products, categories, regions)
  * line: trends over time (daily/monthly sales, growth)
  * pie: proportions/percentages (market share, distribution)
  * scatter: correlation between two numeric variables
  * heatmap: matrix data (region x month, product x category)
- For pie charts, y_column is not needed
- Use descriptive Spanish titles for charts
- Ensure column names in chart_configs match the SELECT query aliases

Database Schema:
{schema}

User Question: {prompt}

Respond with ONLY the JSON object, no additional text."""

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize LLM service.

        Args:
            api_key: Gemini API key (uses settings if not provided)
            model_name: Gemini model name (uses settings if not provided)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model

        # Configure Gemini API
        genai.configure(api_key=self.api_key)

        # Initialize model with configuration
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.2,  # More deterministic
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )

        logger.info(f"LLM service initialized with model: {self.model_name}")

    async def generate_analysis(
        self,
        user_prompt: str,
        database_schema: str,
        additional_context: Optional[str] = None
    ) -> LLMAnalysisResponse:
        """
        Generate SQL query and analysis from natural language prompt.

        Args:
            user_prompt: User's question in Spanish
            database_schema: Formatted database schema
            additional_context: Optional additional context

        Returns:
            Structured LLM response with query, explanation, and chart configs

        Raises:
            LLMAPIError: If API call fails
            LLMParseError: If response cannot be parsed
            LLMValidationError: If response fails validation
        """
        try:
            # Build prompt context
            context = LLMPromptContext(
                database_schema=database_schema,
                user_prompt=user_prompt,
                additional_context=additional_context
            )

            # Generate prompt
            prompt = self._build_prompt(context)

            logger.info(f"Generating analysis for prompt: {user_prompt[:100]}...")

            # Call Gemini API
            response = await self._call_gemini_api(prompt)

            # Parse and validate response
            analysis_response = self._parse_response(response)

            logger.info(
                f"Analysis generated successfully. "
                f"Query: {analysis_response.sql_query[:100]}..."
            )

            return analysis_response

        except LLMAPIError:
            raise
        except LLMParseError:
            raise
        except LLMValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_analysis: {e}")
            raise LLMAPIError(f"Failed to generate analysis: {e}")

    def _build_prompt(self, context: LLMPromptContext) -> str:
        """
        Build complete prompt for Gemini.

        Args:
            context: Prompt context with schema and user question

        Returns:
            Formatted prompt string
        """
        prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            schema=context.database_schema,
            prompt=context.user_prompt
        )

        if context.additional_context:
            prompt += f"\n\nAdditional Context: {context.additional_context}"

        return prompt

    async def _call_gemini_api(self, prompt: str) -> str:
        """
        Call Gemini API with retry logic.

        Args:
            prompt: Complete prompt string

        Returns:
            Raw response text from Gemini

        Raises:
            LLMAPIError: If API call fails after retries
        """
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            try:
                logger.debug(f"Calling Gemini API (attempt {retry_count + 1}/{max_retries + 1})...")

                # Generate content
                response = self.model.generate_content(prompt)

                # Check if response was blocked
                if not response.text:
                    if response.prompt_feedback:
                        logger.error(f"Prompt blocked: {response.prompt_feedback}")
                        raise LLMAPIError(
                            f"Prompt was blocked by safety filters: {response.prompt_feedback}"
                        )
                    raise LLMAPIError("Empty response from Gemini API")

                logger.debug(f"Received response from Gemini ({len(response.text)} chars)")
                return response.text

            except Exception as e:
                retry_count += 1
                logger.warning(f"Gemini API call failed (attempt {retry_count}): {e}")

                if retry_count > max_retries:
                    logger.error(f"Gemini API call failed after {max_retries} retries")
                    raise LLMAPIError(f"Gemini API call failed: {e}")

        raise LLMAPIError("Failed to call Gemini API after retries")

    def _parse_response(self, response_text: str) -> LLMAnalysisResponse:
        """
        Parse and validate LLM response.

        Args:
            response_text: Raw response text from Gemini

        Returns:
            Validated LLMAnalysisResponse object

        Raises:
            LLMParseError: If response cannot be parsed as JSON
            LLMValidationError: If response fails Pydantic validation
        """
        try:
            # Clean response text (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            # Parse JSON
            try:
                response_data = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Response text: {cleaned_text[:500]}")
                raise LLMParseError(
                    f"Failed to parse LLM response as JSON: {e}"
                )

            # Validate with Pydantic
            try:
                analysis_response = LLMAnalysisResponse(**response_data)
                return analysis_response
            except Exception as e:
                logger.error(f"Response validation failed: {e}")
                logger.error(f"Response data: {response_data}")
                raise LLMValidationError(
                    f"LLM response failed validation: {e}"
                )

        except LLMParseError:
            raise
        except LLMValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise LLMParseError(f"Failed to parse response: {e}")

    def test_connection(self) -> bool:
        """
        Test Gemini API connectivity with a simple prompt.

        Returns:
            True if successful, False otherwise
        """
        try:
            test_prompt = "Respond with only the number 1"
            response = self.model.generate_content(test_prompt)
            logger.info("Gemini API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False
