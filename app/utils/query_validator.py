"""
SQL query validator to prevent dangerous operations.
Ensures only safe SELECT queries are executed.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when a query fails validation checks."""
    pass


class QueryValidator:
    """
    Validates SQL queries for safety before execution.

    Blocks:
    - DDL commands (DROP, CREATE, ALTER, TRUNCATE)
    - DML commands except SELECT (INSERT, UPDATE, DELETE)
    - Multiple statements (semicolon-separated)
    - System stored procedures
    - Queries without WHERE clause for UPDATE/DELETE
    """

    # Dangerous SQL keywords that should never be allowed
    BLOCKED_KEYWORDS = [
        r'\bDROP\b',
        r'\bCREATE\b',
        r'\bALTER\b',
        r'\bTRUNCATE\b',
        r'\bINSERT\b',
        r'\bUPDATE\b',
        r'\bDELETE\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bsp_\w+',  # System stored procedures
        r'\bxp_\w+',  # Extended stored procedures
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bDENY\b',
    ]

    # Allowed keywords
    ALLOWED_KEYWORDS = [
        r'\bSELECT\b',
        r'\bFROM\b',
        r'\bWHERE\b',
        r'\bJOIN\b',
        r'\bLEFT\b',
        r'\bRIGHT\b',
        r'\bINNER\b',
        r'\bOUTER\b',
        r'\bON\b',
        r'\bGROUP\s+BY\b',
        r'\bORDER\s+BY\b',
        r'\bHAVING\b',
        r'\bTOP\b',
        r'\bDISTINCT\b',
        r'\bAS\b',
        r'\bCOUNT\b',
        r'\bSUM\b',
        r'\bAVG\b',
        r'\bMAX\b',
        r'\bMIN\b',
        r'\bCASE\b',
        r'\bWHEN\b',
        r'\bTHEN\b',
        r'\bELSE\b',
        r'\bEND\b',
        r'\bAND\b',
        r'\bOR\b',
        r'\bNOT\b',
        r'\bIN\b',
        r'\bLIKE\b',
        r'\bBETWEEN\b',
        r'\bIS\b',
        r'\bNULL\b',
        r'\bDATEADD\b',
        r'\bDATEDIFF\b',
        r'\bGETDATE\b',
        r'\bCONVERT\b',
        r'\bCAST\b',
    ]

    def __init__(self, max_rows: int = 10000):
        """
        Initialize validator.

        Args:
            max_rows: Maximum number of rows to allow in results
        """
        self.max_rows = max_rows

    def validate(self, query: str) -> None:
        """
        Validate a SQL query for safety.

        Args:
            query: SQL query string to validate

        Raises:
            QueryValidationError: If query fails validation

        Returns:
            None if query is valid
        """
        if not query or not query.strip():
            raise QueryValidationError("Query cannot be empty")

        # Normalize query for checking
        normalized_query = query.upper().strip()

        # Check 1: Must start with SELECT
        if not self._starts_with_select(normalized_query):
            raise QueryValidationError(
                "Only SELECT queries are allowed. Query must start with SELECT."
            )

        # Check 2: Block dangerous keywords
        self._check_blocked_keywords(normalized_query)

        # Check 3: Block multiple statements (semicolons)
        self._check_multiple_statements(query)

        # Check 4: Block comment-based SQL injection attempts
        self._check_comments(query)

        # Check 5: Ensure TOP clause or add warning if missing
        self._check_row_limit(normalized_query)
        self._check_clients_id_only(query)

        logger.info(f"Query validated successfully: {query[:100]}...")

    def _starts_with_select(self, query: str) -> bool:
        """Check if query starts with SELECT."""
        # Remove leading whitespace and comments
        query = re.sub(r'^\s*(--.*?\n|/\*.*?\*/)*\s*', '', query, flags=re.DOTALL)
        return query.startswith('SELECT')

    def _check_blocked_keywords(self, query: str) -> None:
        """Check for blocked SQL keywords."""
        for pattern in self.BLOCKED_KEYWORDS:
            if re.search(pattern, query, re.IGNORECASE):
                keyword = re.search(pattern, query, re.IGNORECASE).group(0)
                raise QueryValidationError(
                    f"Blocked keyword detected: {keyword}. "
                    f"Only SELECT queries are allowed."
                )

    def _check_multiple_statements(self, query: str) -> None:
        """Block multiple statements (semicolon-separated)."""
        # Remove string literals to avoid false positives
        query_without_strings = re.sub(r"'[^']*'", "", query)

        # Count semicolons (allow one at the end)
        semicolons = query_without_strings.count(';')
        if semicolons > 1 or (semicolons == 1 and not query.strip().endswith(';')):
            raise QueryValidationError(
                "Multiple statements are not allowed. "
                "Only single SELECT queries are permitted."
            )

    def _check_comments(self, query: str) -> None:
        """Check for suspicious comment patterns (SQL injection)."""
        # Block -- comments that might be used for injection
        if re.search(r'--', query):
            logger.warning(f"Query contains -- comments: {query[:100]}")
            # Allow but log - comments can be legitimate

        # Block /* */ comments
        if re.search(r'/\*.*?\*/', query, re.DOTALL):
            logger.warning(f"Query contains /* */ comments: {query[:100]}")

    def _check_row_limit(self, query: str) -> None:
        """Check if query has TOP clause."""
        if 'TOP' not in query:
            logger.warning(
                f"Query does not contain TOP clause. "
                f"Results may exceed {self.max_rows} rows."
            )
        else:
            # Extract TOP value
            top_match = re.search(r'TOP\s+(\d+)', query)
            if top_match:
                top_value = int(top_match.group(1))
                if top_value > self.max_rows:
                    raise QueryValidationError(
                        f"TOP value ({top_value}) exceeds maximum "
                        f"allowed rows ({self.max_rows})"
                    )

    def _check_clients_id_only(self, query: str) -> None:
        if not re.search(r"\bFROM\s+Clients\b|\bJOIN\s+Clients\b", query, re.IGNORECASE):
            return

        select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return

        select_list = select_match.group(1)
        if re.search(r"\*", select_list):
            raise QueryValidationError(
                "Clients table cannot be selected with '*'. Only Id is allowed."
            )

        disallowed = re.findall(
            r"\bClients\.([A-Za-z_][A-Za-z0-9_]*)\b", select_list, re.IGNORECASE
        )
        for col in disallowed:
            if col.lower() != "id":
                raise QueryValidationError(
                    "Only Clients.Id can be selected from Clients."
                )

        qualified_stripped = re.sub(
            r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b",
            " ",
            select_list,
        )
        bare_cols = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", qualified_stripped)
        for col in bare_cols:
            if col.lower() not in {"select", "distinct", "top", "id", "as"}:
                raise QueryValidationError(
                    "Only Id can be selected when querying Clients."
                )

        where_on = re.findall(
            r"\b(WHERE|ON)\b(.*?)(\bGROUP\b|\bORDER\b|\bHAVING\b|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        for _, clause, _ in where_on:
            for col in re.findall(
                r"\bClients\.([A-Za-z_][A-Za-z0-9_]*)\b", clause, re.IGNORECASE
            ):
                if col.lower() != "id":
                    raise QueryValidationError(
                        "Only Clients.Id can be used in WHERE/JOIN conditions."
                    )

    def sanitize_query(self, query: str) -> str:
        """
        Sanitize and format query.
        Removes unnecessary whitespace and normalizes format.

        Args:
            query: Raw SQL query

        Returns:
            Sanitized query string
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        # Remove trailing semicolon if present
        query = query.rstrip(';')

        return query

    def add_row_limit(self, query: str, limit: int = None) -> str:
        """
        Add TOP clause to query if not present.

        Args:
            query: SQL query
            limit: Row limit (uses max_rows if not specified)

        Returns:
            Query with TOP clause
        """
        limit = limit or self.max_rows
        normalized = query.upper().strip()

        if 'TOP' in normalized:
            return query  # Already has TOP clause

        # Insert TOP after SELECT
        query = re.sub(
            r'(SELECT)\s+',
            f'SELECT TOP {limit} ',
            query,
            count=1,
            flags=re.IGNORECASE
        )

        return query
