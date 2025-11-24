"""
Database repository for SQL Server access.
Handles connection management and query execution.
"""

import logging
from typing import List, Dict, Any
import pyodbc
from contextlib import contextmanager

from app.core.config import settings
from app.utils.db_schema_loader import SchemaLoader
from app.utils.query_validator import QueryValidator, QueryValidationError

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseRepository:
    """
    Repository for SQL Server database operations.
    Provides safe query execution with validation.
    """

    def __init__(self, connection_string: str = None, max_rows: int = None):
        """
        Initialize database repository.

        Args:
            connection_string: SQL Server connection string (uses settings if not provided)
            max_rows: Maximum rows to return (uses settings if not provided)
        """
        self.connection_string = connection_string or settings.db_connection_string
        self.max_rows = max_rows or settings.max_query_rows
        self.query_timeout = settings.query_timeout_seconds
        self.schema_loader = SchemaLoader(self.connection_string)
        self.query_validator = QueryValidator(max_rows=self.max_rows)

    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed.

        Yields:
            pyodbc.Connection object

        Raises:
            DatabaseConnectionError: If connection fails
        """
        conn = None
        try:
            logger.debug("Establishing database connection...")
            conn = pyodbc.connect(
                self.connection_string,
                timeout=self.query_timeout
            )
            conn.timeout = self.query_timeout
            logger.debug("Database connection established")
            yield conn
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")

    def test_connection(self) -> bool:
        """
        Test database connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_schema(self, exclude_tables: List[str] = None) -> str:
        """
        Get formatted database schema for LLM context.

        Args:
            exclude_tables: List of tables to exclude (e.g., sensitive data)

        Returns:
            Formatted schema string

        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            logger.info("Loading database schema...")
            schema = self.schema_loader.get_schema(exclude_tables)
            logger.info(f"Schema loaded successfully ({len(schema)} characters)")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise DatabaseConnectionError(f"Failed to load database schema: {e}")

    def execute_query(self, query: str, validate: bool = True) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dictionaries.

        Args:
            query: SQL query to execute
            validate: Whether to validate query for safety (default: True)

        Returns:
            List of dictionaries with column names as keys

        Raises:
            QueryValidationError: If query validation fails
            DatabaseConnectionError: If query execution fails
        """
        # Validate query
        if validate:
            try:
                self.query_validator.validate(query)
                query = self.query_validator.sanitize_query(query)

                # Add TOP clause if missing
                if 'TOP' not in query.upper():
                    query = self.query_validator.add_row_limit(query, self.max_rows)
                    logger.info(f"Added TOP {self.max_rows} clause to query")
            except QueryValidationError as e:
                logger.error(f"Query validation failed: {e}")
                raise

        # Execute query
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                logger.info(f"Executing query: {query[:200]}...")
                cursor.execute(query)

                # Fetch column names
                columns = [column[0] for column in cursor.description]

                # Fetch all rows
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                results = []
                for row in rows:
                    row_dict = {}
                    for i, value in enumerate(row):
                        # Convert pyodbc types to Python types
                        if value is None:
                            row_dict[columns[i]] = None
                        elif isinstance(value, (int, float, str, bool)):
                            row_dict[columns[i]] = value
                        else:
                            # Convert other types to string
                            row_dict[columns[i]] = str(value)
                    results.append(row_dict)

                cursor.close()

                logger.info(f"Query executed successfully. Returned {len(results)} rows.")
                return results

        except pyodbc.Error as e:
            logger.error(f"Query execution error: {e}")
            raise DatabaseConnectionError(f"Failed to execute query: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {e}")
            raise

    def execute_query_raw(self, query: str) -> List[tuple]:
        """
        Execute query and return raw results (tuples).
        For internal use - bypasses validation.

        Args:
            query: SQL query

        Returns:
            List of tuples

        Raises:
            DatabaseConnectionError: If execution fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                cursor.close()
                return results
        except Exception as e:
            logger.error(f"Raw query execution failed: {e}")
            raise DatabaseConnectionError(f"Query execution failed: {e}")

    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database.

        Returns:
            List of table names
        """
        return self.schema_loader.get_table_names()

    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample rows from a table.
        Useful for understanding data format.

        Args:
            table_name: Name of the table
            limit: Number of rows to return

        Returns:
            List of dictionaries with sample data
        """
        query = f"SELECT TOP {limit} * FROM {table_name}"
        return self.execute_query(query, validate=False)

    def get_column_stats(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific column.

        Args:
            table_name: Table name
            column_name: Column name

        Returns:
            Dictionary with statistics (count, min, max, etc.)
        """
        query = f"""
            SELECT
                COUNT(*) as total_count,
                COUNT({column_name}) as non_null_count,
                COUNT(DISTINCT {column_name}) as distinct_count
            FROM {table_name}
        """

        try:
            results = self.execute_query(query, validate=False)
            return results[0] if results else {}
        except Exception as e:
            logger.error(f"Failed to get column stats: {e}")
            return {}
