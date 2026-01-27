"""
LLM service for Google Gemini API integration.
Generates SQL queries and analysis from natural language prompts.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

from app.core.config import settings
from app.models.llm_models import (
    LLMMultiQueryResponse,
    QueryPlan,
    LLMPromptContext,
    LLMAPIError,
    LLMParseError,
    LLMValidationError
)

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Google Gemini LLM.
    Generates multi-query analysis plans from natural language prompts.
    """

    SYSTEM_PROMPT_MULTI_QUERY = """You are an expert SQL analyst for a business database. Given a database schema and a user question in Spanish, you must generate 1-5 SQL queries to comprehensively answer the question.

Generate multiple queries when:
- The question has multiple parts (e.g., "top products AND monthly trends")
- Comparison data is needed (e.g., "this month vs last month")
- Different perspectives add value (e.g., "sales by category AND by region")
- Supporting context improves the answer

Generate a single query when:
- The question is straightforward
- One query fully answers the question

Respond ONLY with valid JSON matching this exact structure:
{{
  "queries": [
    {{
      "query_id": "q1",
      "purpose": "Obtener los productos mas vendidos",
      "sql_query": "SELECT TOP 10 p.Name as ProductName, SUM(s.Quantity) as TotalSales FROM Sales s JOIN Products p ON s.ProductId = p.Id GROUP BY p.Name ORDER BY TotalSales DESC"
    }},
    {{
      "query_id": "q2",
      "purpose": "Comparar con el mes anterior",
      "sql_query": "SELECT SUM(s.Quantity) as LastMonthSales FROM Sales s WHERE s.SaleDate >= DATEADD(month, -1, DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)) AND s.SaleDate < DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)"
    }}
  ]
}}

CRITICAL SQL RULES:
- Use SQL Server syntax (TOP instead of LIMIT, GETDATE() instead of NOW())
- Always include column aliases for aggregates (e.g., SUM(quantity) as TotalSales)
- **GROUP BY RULE**: When using aggregate functions (SUM, COUNT, AVG, MAX, MIN), ALL non-aggregated columns in SELECT must be in GROUP BY
  * CORRECT: SELECT ProductName, SUM(Quantity) as Total FROM Sales GROUP BY ProductName
  * WRONG: SELECT ProductName, CategoryName, SUM(Quantity) as Total FROM Sales GROUP BY ProductName
  * CORRECT: SELECT ProductName, CategoryName, SUM(Quantity) as Total FROM Sales GROUP BY ProductName, CategoryName
- Entity Framework databases use PascalCase column names (e.g., ProductId, CategoryName, CreatedDate)
- When joining tables, always use aliases to avoid ambiguous column references
  * Example: SELECT p.Name, SUM(s.Quantity) as Total FROM Sales s JOIN Products p ON s.ProductId = p.ProductId GROUP BY p.Name

QUERY RULES:
- Generate 1-5 queries maximum
- Each query must have a unique query_id (q1, q2, q3, etc.)
- Each purpose must be in Spanish and describe what that specific query answers
- Queries should be complementary - together they should fully answer the user's question
- Order queries logically (main query first, supporting/comparison queries after)
- JSON strings must be valid: do NOT include raw newlines inside string values
  * If you need line breaks in sql_query, use \\n inside the string
  * Ensure the JSON is complete and strictly valid

Database Schema:
{schema}

User Question: {prompt}

Maximum queries allowed: {max_queries}

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
            global_context: Optional global context for prompts
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
            max_output_tokens=4096,
            response_mime_type="application/json",
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

    async def generate_multi_query_plan(
        self,
        user_prompt: str,
        database_schema: str,
        max_queries: int = 5,
        additional_context: Optional[str] = None
    ) -> LLMMultiQueryResponse:
        """
        Generate a multi-query plan from natural language prompt.

        Args:
            user_prompt: User's question in Spanish
            database_schema: Formatted database schema
            max_queries: Maximum number of queries to generate (1-5)
            additional_context: Optional additional context

        Returns:
            Structured LLM response with query plan

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
            prompt = self._build_multi_query_prompt(context, max_queries)

            logger.info(
                f"Generating multi-query plan for prompt: {user_prompt[:100]}... "
                f"(max_queries: {max_queries})"
            )

            # Call Gemini API
            response = await self._call_gemini_api(prompt)

            # Parse and validate response
            multi_query_response = self._parse_multi_query_response(response)

            logger.info(
                f"Multi-query plan generated successfully. "
                f"Queries: {len(multi_query_response.queries)}"
            )

            return multi_query_response

        except LLMAPIError:
            raise
        except LLMParseError:
            raise
        except LLMValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_multi_query_plan: {e}")
            raise LLMAPIError(f"Failed to generate query plan: {e}")

    def _build_multi_query_prompt(
        self,
        context: LLMPromptContext,
        max_queries: int
    ) -> str:
        """
        Build complete prompt for multi-query generation.

        Args:
            context: Prompt context with schema and user question
            max_queries: Maximum number of queries to generate

        Returns:
            Formatted prompt string
        """
        prompt = self.SYSTEM_PROMPT_MULTI_QUERY.format(
            schema=context.database_schema,
            prompt=context.user_prompt,
            max_queries=max_queries
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

    def _parse_multi_query_response(self, response_text: str) -> LLMMultiQueryResponse:
        """
        Parse and validate LLM response for multi-query plan.

        Args:
            response_text: Raw response text from Gemini

        Returns:
            Validated LLMMultiQueryResponse object

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

                recovered_text = self._extract_json_object(cleaned_text)
                if recovered_text:
                    sanitized_text = self._sanitize_json_string_literals(
                        recovered_text
                    )
                    try:
                        response_data = json.loads(sanitized_text)
                    except json.JSONDecodeError as recovery_error:
                        logger.error(
                            f"Failed to parse sanitized JSON: {recovery_error}"
                        )
                        raise LLMParseError(
                            f"Failed to parse LLM response as JSON: {e}"
                        ) from recovery_error
                else:
                    raise LLMParseError(
                        f"Failed to parse LLM response as JSON: {e}"
                    )

            # Validate with Pydantic
            try:
                multi_query_response = LLMMultiQueryResponse(**response_data)
                return multi_query_response
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

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """
        Extract a JSON object from a larger string.

        Returns None if no object boundaries are found.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1]

    @staticmethod
    def _sanitize_json_string_literals(text: str) -> str:
        """
        Escape raw control characters inside JSON string literals.

        This makes LLM output with unescaped newlines parseable.
        """
        result = []
        in_string = False
        escape = False

        for ch in text:
            if in_string:
                if escape:
                    if ch == "\r":
                        result.append("r")
                    elif ch == "\n":
                        result.append("n")
                    elif ch == "\t":
                        result.append("t")
                    else:
                        result.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    result.append(ch)
                    escape = True
                    continue
                if ch == "\"":
                    result.append(ch)
                    in_string = False
                    continue
                if ch == "\r":
                    result.append("\\r")
                    continue
                if ch == "\n":
                    result.append("\\n")
                    continue
                if ch == "\t":
                    result.append("\\t")
                    continue
                result.append(ch)
            else:
                if ch == "\"":
                    result.append(ch)
                    in_string = True
                else:
                    result.append(ch)

        return "".join(result)

    async def generate_unified_analysis(
        self,
        user_prompt: str,
        query_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate unified analysis based on all query results.

        Args:
            user_prompt: Original user question in Spanish
            query_results: List of query results with their data

        Returns:
            Unified analysis text in Spanish synthesizing all results

        Raises:
            LLMAPIError: If API call fails
        """
        try:
            # Format query results for the prompt
            results_text = ""
            for result in query_results:
                query_id = result.get("query_id", "unknown")
                purpose = result.get("purpose", "")
                data = result.get("data", [])
                error = result.get("error")
                row_count = result.get("row_count", 0)

                results_text += f"\n--- Query {query_id}: {purpose} ---\n"
                if error:
                    results_text += f"Error: {error}\n"
                else:
                    # Limit data to first 30 rows for context
                    sample_data = data[:30]
                    results_text += f"Rows returned: {row_count}\n"
                    results_text += f"Data (first 30 rows):\n{json.dumps(sample_data, indent=2, ensure_ascii=False)}\n"

            global_context_block = ""
            if self.global_context:
                global_context_block = f"\n\nGlobal Context:\n{self.global_context}"

            # Build prompt for unified analysis
            analysis_prompt = f"""You are an expert data analyst. Given a user's question and the results from multiple SQL queries, provide a unified analysis in Spanish that synthesizes all the data.

User Question: {user_prompt}

Query Results:
{results_text}
{global_context_block}

Provide a comprehensive analysis in Spanish (3-8 sentences) that:
1. Directly answers the user's question by synthesizing data from ALL successful queries
2. Highlights key findings, comparisons, and patterns across the data
3. Uses specific numbers and values from the results
4. If some queries failed, acknowledge this but focus on available data
5. Is written in a professional but accessible tone

Respond with ONLY the analysis text in Spanish, no JSON or additional formatting."""

            logger.info(f"Generating unified analysis from {len(query_results)} query results...")

            # Call Gemini API
            response = await self._call_gemini_api(analysis_prompt)

            # Clean and return analysis
            analysis = response.strip()

            logger.info(f"Generated unified analysis: {analysis[:100]}...")

            return analysis

        except Exception as e:
            logger.error(f"Failed to generate unified analysis: {e}")
            # Return a basic fallback analysis
            successful = sum(1 for r in query_results if not r.get("error"))
            total_rows = sum(r.get("row_count", 0) for r in query_results if not r.get("error"))
            return f"Se ejecutaron {successful} consultas exitosamente con un total de {total_rows} registros."

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
