"""
Database schema introspection utilities.
Loads and formats database schema for LLM context.
"""

import logging
from typing import Dict, List
import pyodbc

logger = logging.getLogger(__name__)


class SchemaLoader:
    """
    Loads database schema information for SQL Server.
    Provides formatted schema for LLM context.
    """

    def __init__(self, connection_string: str):
        """
        Initialize schema loader.

        Args:
            connection_string: SQL Server connection string
        """
        self.connection_string = connection_string

    def get_schema(self, exclude_tables: List[str] = None) -> str:
        """
        Get formatted database schema for LLM context.

        Args:
            exclude_tables: List of table names to exclude (e.g., sensitive tables)

        Returns:
            Formatted schema string for LLM prompt
        """
        exclude_tables = exclude_tables or []

        try:
            tables = self._get_tables_info(exclude_tables)
            return self._format_schema(tables)
        except Exception as e:
            logger.error(f"Failed to load database schema: {e}")
            raise

    def _get_tables_info(self, exclude_tables: List[str]) -> Dict:
        """
        Query database for schema information.

        Args:
            exclude_tables: Tables to exclude

        Returns:
            Dictionary with table information
        """
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()

        tables_info = {}

        try:
            # Get all user tables
            tables_query = """
                SELECT
                    t.TABLE_NAME,
                    t.TABLE_TYPE
                FROM INFORMATION_SCHEMA.TABLES t
                WHERE t.TABLE_TYPE = 'BASE TABLE'
                    AND t.TABLE_SCHEMA = 'dbo'
                ORDER BY t.TABLE_NAME
            """

            cursor.execute(tables_query)
            tables = cursor.fetchall()

            for table in tables:
                table_name = table.TABLE_NAME

                # Skip excluded tables
                if table_name in exclude_tables:
                    continue

                # Get columns for this table
                columns_query = """
                    SELECT
                        c.COLUMN_NAME,
                        c.DATA_TYPE,
                        c.CHARACTER_MAXIMUM_LENGTH,
                        c.IS_NULLABLE,
                        c.COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS c
                    WHERE c.TABLE_NAME = ?
                        AND c.TABLE_SCHEMA = 'dbo'
                    ORDER BY c.ORDINAL_POSITION
                """

                cursor.execute(columns_query, table_name)
                columns = cursor.fetchall()

                # Get primary keys
                pk_query = """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
                        AND TABLE_NAME = ?
                        AND TABLE_SCHEMA = 'dbo'
                """

                cursor.execute(pk_query, table_name)
                primary_keys = [row.COLUMN_NAME for row in cursor.fetchall()]

                # Get foreign keys
                fk_query = """
                    SELECT
                        fk.name AS FK_NAME,
                        COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS COLUMN_NAME,
                        OBJECT_NAME(fkc.referenced_object_id) AS REFERENCED_TABLE,
                        COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS REFERENCED_COLUMN
                    FROM sys.foreign_keys fk
                    INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                    WHERE OBJECT_NAME(fk.parent_object_id) = ?
                """

                cursor.execute(fk_query, table_name)
                foreign_keys = cursor.fetchall()

                tables_info[table_name] = {
                    'columns': columns,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys
                }

        finally:
            cursor.close()
            conn.close()

        return tables_info

    def _format_schema(self, tables_info: Dict) -> str:
        """
        Format schema information for LLM context.

        Args:
            tables_info: Dictionary with table information

        Returns:
            Formatted schema string
        """
        schema_lines = ["Database Schema:", ""]

        for table_name, info in tables_info.items():
            schema_lines.append(f"Table: {table_name}")

            # Add columns
            for col in info['columns']:
                col_name = col.COLUMN_NAME
                col_type = col.DATA_TYPE

                # Add length for string types
                if col.CHARACTER_MAXIMUM_LENGTH:
                    col_type += f"({col.CHARACTER_MAXIMUM_LENGTH})"

                # Build column description
                col_desc = f"- {col_name} ({col_type}"

                # Add constraints
                constraints = []
                if col_name in info['primary_keys']:
                    constraints.append("PRIMARY KEY")

                # Check for foreign key
                for fk in info['foreign_keys']:
                    if fk.COLUMN_NAME == col_name:
                        constraints.append(
                            f"FOREIGN KEY -> {fk.REFERENCED_TABLE}.{fk.REFERENCED_COLUMN}"
                        )

                if col.IS_NULLABLE == 'NO':
                    constraints.append("NOT NULL")

                if constraints:
                    col_desc += ", " + ", ".join(constraints)

                col_desc += ")"
                schema_lines.append(col_desc)

            schema_lines.append("")  # Empty line between tables

        return "\n".join(schema_lines)

    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database.

        Returns:
            List of table names
        """
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                    AND TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """)

            return [row.TABLE_NAME for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

    def get_table_row_count(self, table_name: str) -> int:
        """
        Get approximate row count for a table.

        Args:
            table_name: Name of the table

        Returns:
            Approximate number of rows
        """
        conn = pyodbc.connect(self.connection_string)
        cursor = conn.cursor()

        try:
            # Use sys.partitions for quick estimate
            cursor.execute("""
                SELECT SUM(p.rows) AS row_count
                FROM sys.partitions p
                INNER JOIN sys.objects o ON p.object_id = o.object_id
                WHERE o.name = ?
                    AND p.index_id IN (0, 1)
            """, table_name)

            result = cursor.fetchone()
            return result.row_count if result else 0
        finally:
            cursor.close()
            conn.close()
