"""
LLM service for Google Gemini API integration.
Generates SQL queries and analysis from natural language prompts.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types

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

    SYSTEM_PROMPT_WITH_CHARTS = """You are an expert SQL analyst for a business database. Given a database schema and a user question in Spanish, you must:

1. Write a SQL Server query to answer the question
2. Leave explanation empty (will be generated after query execution)
3. Suggest appropriate chart visualizations

Respond ONLY with valid JSON matching this exact structure:
{{
  "sql_query": "SELECT TOP 10 p.Name as ProductName, SUM(s.Quantity) as TotalSales FROM Sales s JOIN Products p ON s.ProductId = p.Id GROUP BY p.Name ORDER BY TotalSales DESC",
  "explanation": "",
  "chart_configs": [
    {{
      "type": "bar",
      "title": "Top 10 Productos Más Vendidos",
      "x_column": "ProductName",
      "y_column": "TotalSales",
      "x_label": "Producto",
      "y_label": "Ventas Totales",
      "color_palette": "viridis"
    }}
  ]
}}

CRITICAL SQL RULES:
- Use SQL Server syntax (TOP instead of LIMIT, GETDATE() instead of NOW())
- Always include column aliases for aggregates (e.g., SUM(quantity) as total_sales)
- **GROUP BY RULE**: When using aggregate functions (SUM, COUNT, AVG, MAX, MIN), ALL non-aggregated columns in SELECT must be in GROUP BY
  * CORRECT: SELECT ProductName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * WRONG: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * CORRECT: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName, CategoryName
- Entity Framework databases use PascalCase column names (e.g., ProductId, CategoryName, CreatedDate)
- When joining tables, always use aliases to avoid ambiguous column references
  * Example: SELECT p.Name, SUM(s.Quantity) as total FROM Sales s JOIN Products p ON s.ProductId = p.ProductId GROUP BY p.Name
- Explanation must be in Spanish, clear and concise

CHART RULES:
- Maximum 3 charts per analysis
- Choose chart types appropriate for the data:
  * bar: categorical comparisons (products, categories, regions)
  * line: trends over time (daily/monthly sales, growth)
  * pie: proportions/percentages (market share, distribution)
  * scatter: correlation between two numeric variables
  * heatmap: matrix data (region x month, product x category)
- For pie charts, y_column is not needed
- Use descriptive Spanish titles for charts
- Ensure column names in chart_configs match the SELECT query aliases EXACTLY

Database Schema:
{schema}

User Question: {prompt}

Respond with ONLY the JSON object, no additional text."""

    SYSTEM_PROMPT_WITHOUT_CHARTS = """You are an expert SQL analyst for a business database. Given a database schema and a user question in Spanish, you must:

1. Write a SQL Server query to answer the question
2. Leave explanation empty (will be generated after query execution)
3. Do NOT generate any chart visualizations

Respond ONLY with valid JSON matching this exact structure:
{{
  "sql_query": "SELECT TOP 10 p.Name as ProductName, SUM(s.Quantity) as TotalSales FROM Sales s JOIN Products p ON s.ProductId = p.Id GROUP BY p.Name ORDER BY TotalSales DESC",
  "explanation": "",
  "chart_configs": []
}}

CRITICAL SQL RULES:
- Use SQL Server syntax (TOP instead of LIMIT, GETDATE() instead of NOW())
- Always include column aliases for aggregates (e.g., SUM(quantity) as total_sales)
- **GROUP BY RULE**: When using aggregate functions (SUM, COUNT, AVG, MAX, MIN), ALL non-aggregated columns in SELECT must be in GROUP BY
  * CORRECT: SELECT ProductName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * WRONG: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * CORRECT: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName, CategoryName
- Entity Framework databases use PascalCase column names (e.g., ProductId, CategoryName, CreatedDate)
- When joining tables, always use aliases to avoid ambiguous column references
  * Example: SELECT p.Name, SUM(s.Quantity) as total FROM Sales s JOIN Products p ON s.ProductId = p.ProductId GROUP BY p.Name
- Explanation must be in Spanish, clear and concise
- Do NOT include any charts - the chart_configs array must be empty: []
- Focus on providing a detailed textual explanation since no visualizations will be provided

Database Schema:
{schema}

User Question: {prompt}

Respond with ONLY the JSON object, no additional text."""

    SYSTEM_PROMPT_AUTO_CHARTS = """You are an expert SQL analyst for a business database. Given a database schema and a user question in Spanish, you must:

1. Write a SQL Server query to answer the question
2. Leave explanation empty (will be generated after query execution)
3. Decide whether chart visualizations would be helpful for understanding the data
4. Only suggest charts if they add significant value to the analysis

Respond ONLY with valid JSON matching this exact structure:
{{
  "sql_query": "SELECT TOP 10 p.Name as ProductName, SUM(s.Quantity) as TotalSales FROM Sales s JOIN Products p ON s.ProductId = p.Id GROUP BY p.Name ORDER BY TotalSales DESC",
  "explanation": "",
  "chart_configs": [
    {{
      "type": "bar",
      "title": "Top 10 Productos Más Vendidos",
      "x_column": "ProductName",
      "y_column": "TotalSales",
      "x_label": "Producto",
      "y_label": "Ventas Totales",
      "color_palette": "viridis"
    }}
  ]
}}

CRITICAL SQL RULES:
- Use SQL Server syntax (TOP instead of LIMIT, GETDATE() instead of NOW())
- Always include column aliases for aggregates (e.g., SUM(quantity) as total_sales)
- **GROUP BY RULE**: When using aggregate functions (SUM, COUNT, AVG, MAX, MIN), ALL non-aggregated columns in SELECT must be in GROUP BY
  * CORRECT: SELECT ProductName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * WRONG: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName
  * CORRECT: SELECT ProductName, CategoryName, SUM(Quantity) as total FROM Sales GROUP BY ProductName, CategoryName
- Entity Framework databases use PascalCase column names (e.g., ProductId, CategoryName, CreatedDate)
- When joining tables, always use aliases to avoid ambiguous column references
  * Example: SELECT p.Name, SUM(s.Quantity) as total FROM Sales s JOIN Products p ON s.ProductId = p.ProductId GROUP BY p.Name
- Explanation must be in Spanish, clear and concise

CHART RULES:
- Only suggest charts when they genuinely help understand the data:
  * YES: Comparisons, trends, distributions, correlations
  * NO: Simple counts, single values, yes/no answers, basic lookups
- Maximum 3 charts per analysis (or use empty array [] if charts aren't needed)
- Choose chart types appropriate for the data:
  * bar: categorical comparisons (products, categories, regions)
  * line: trends over time (daily/monthly sales, growth)
  * pie: proportions/percentages (market share, distribution)
  * scatter: correlation between two numeric variables
  * heatmap: matrix data (region x month, product x category)
- For pie charts, y_column is not needed
- Use descriptive Spanish titles for charts
- Ensure column names in chart_configs match the SELECT query aliases EXACTLY

Database Schema:
{schema}

User Question: {prompt}

Respond with ONLY the JSON object, no additional text."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        global_context: Optional[str] = None
    ):
        """
        Initialize LLM service.

        Args:
            api_key: Gemini API key (uses settings if not provided)
            model_name: Gemini model name (uses settings if not provided)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.global_context = (
            global_context
            if global_context is not None
            else self._resolve_global_context()
        )

        # Initialize Gemini client with API key
        self.client = genai.Client(api_key=self.api_key)

        # Configure generation settings
        self.generation_config = types.GenerateContentConfig(
            temperature=0.2,  # More deterministic
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            safety_settings=[
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
            ]
        )

        logger.info(f"LLM service initialized with model: {self.model_name}")

    async def generate_analysis(
        self,
        user_prompt: str,
        database_schema: str,
        additional_context: Optional[str] = None,
        generate_charts: Optional[bool] = None
    ) -> LLMAnalysisResponse:
        """
        Generate SQL query and analysis from natural language prompt.

        Args:
            user_prompt: User's question in Spanish
            database_schema: Formatted database schema
            additional_context: Optional additional context
            generate_charts: Whether to generate charts (None=auto, True=always, False=never)

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
            prompt = self._build_prompt(context, generate_charts)

            logger.info(
                f"Generating analysis for prompt: {user_prompt[:100]}... "
                f"(charts: {generate_charts})"
            )

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

    def _build_prompt(
        self,
        context: LLMPromptContext,
        generate_charts: Optional[bool] = None
    ) -> str:
        """
        Build complete prompt for Gemini.

        Args:
            context: Prompt context with schema and user question
            generate_charts: Whether to generate charts (None=auto, True=always, False=never)

        Returns:
            Formatted prompt string
        """
        # Select appropriate template based on chart generation preference
        if generate_charts is True:
            template = self.SYSTEM_PROMPT_WITH_CHARTS
        elif generate_charts is False:
            template = self.SYSTEM_PROMPT_WITHOUT_CHARTS
        else:  # None - let LLM decide
            template = self.SYSTEM_PROMPT_AUTO_CHARTS

        prompt = template.format(
            schema=context.database_schema,
            prompt=context.user_prompt
        )

        if self.global_context:
            prompt += f"\n\nGlobal Context:\n{self.global_context}"

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

                # Generate content using client.models.generate_content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config
                )

                # Check if response has text
                if not response.text:
                    logger.error(f"Empty response from Gemini API")
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

    async def generate_explanation_from_results(
        self,
        user_prompt: str,
        sql_query: str,
        query_results: list
    ) -> str:
        """
        Generate explanation based on actual query results.

        Args:
            user_prompt: Original user question in Spanish
            sql_query: The executed SQL query
            query_results: List of result rows (dicts)

        Returns:
            Explanation text in Spanish describing the results

        Raises:
            LLMAPIError: If API call fails
        """
        try:
            # Limit results to first 50 rows for context (to avoid token limits)
            results_sample = query_results[:50]

            global_context_block = ""
            if self.global_context:
                global_context_block = f"\n\nGlobal Context:\n{self.global_context}"

            # Build prompt for explanation generation
            explanation_prompt = f"""You are an expert data analyst. Given a user's question, the SQL query that was executed, and the actual results, provide a clear, concise explanation in Spanish that describes the RESULTS.

User Question: {user_prompt}

SQL Query Executed:
{sql_query}

Query Results (first 50 rows):
{json.dumps(results_sample, indent=2, ensure_ascii=False)}

Total Rows Returned: {len(query_results)}
{global_context_block}

Provide a clear explanation in Spanish (2-5 sentences) that:
1. Directly answers the user's question based on the results
2. Highlights key findings and patterns in the data
3. Uses specific numbers and values from the results
4. Is written in a professional but accessible tone

Respond with ONLY the explanation text in Spanish, no JSON or additional formatting."""

            logger.info(f"Generating explanation from {len(query_results)} result rows...")

            # Call Gemini API
            response = await self._call_gemini_api(explanation_prompt)

            # Clean and return explanation
            explanation = response.strip()

            logger.info(f"Generated explanation: {explanation[:100]}...")

            return explanation

        except Exception as e:
            logger.error(f"Failed to generate explanation from results: {e}")
            # Return a basic fallback explanation
            return f"Se ejecutó la consulta exitosamente y se obtuvieron {len(query_results)} resultados."

    def test_connection(self) -> bool:
        """
        Test Gemini API connectivity with a simple prompt.

        Returns:
            True if successful, False otherwise
        """
        try:
            test_prompt = "Respond with only the number 1"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=test_prompt
            )
            logger.info("Gemini API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

    def _resolve_global_context(self) -> str:
        """
        Load global LLM context from settings.

        Precedence:
        1. LLM_GLOBAL_CONTEXT_PATH file contents
        2. LLM_GLOBAL_CONTEXT inline value
        """
        context_path = settings.llm_global_context_path.strip()
        if context_path:
            path = Path(context_path)
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(
                    f"Failed to read LLM global context file '{context_path}': {e}"
                )

        return settings.llm_global_context.strip()
