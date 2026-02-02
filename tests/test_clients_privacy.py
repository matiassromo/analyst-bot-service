from types import SimpleNamespace

from contextlib import contextmanager
import pytest

from app.utils.db_schema_loader import SchemaLoader
from app.utils.query_validator import QueryValidationError, QueryValidator
from app.repositories.database_repository import DatabaseRepository


def _col(name, dtype="int", max_len=None, nullable="NO"):
    return SimpleNamespace(
        COLUMN_NAME=name,
        DATA_TYPE=dtype,
        CHARACTER_MAXIMUM_LENGTH=max_len,
        IS_NULLABLE=nullable,
        COLUMN_DEFAULT=None,
    )


def test_schema_filters_clients_to_id_only():
    loader = SchemaLoader(
        "Driver={ODBC Driver 17 for SQL Server};Server=.;Database=Fake;Trusted_Connection=yes;"
    )
    tables_info = {
        "Clients": {
            "columns": [_col("Id"), _col("Name", "nvarchar", 100)],
            "primary_keys": ["Id"],
            "foreign_keys": [],
        },
        "Orders": {
            "columns": [_col("Id"), _col("ClientId")],
            "primary_keys": ["Id"],
            "foreign_keys": [],
        },
    }

    filtered = loader._apply_column_allowlist(tables_info, {"Clients": ["Id"]})
    schema = loader._format_schema(filtered)

    assert "Table: Clients" in schema
    assert "- Id (int, PRIMARY KEY, NOT NULL)" in schema
    assert "Name" not in schema


def test_validator_blocks_non_id_clients_columns():
    validator = QueryValidator()

    validator.validate("SELECT Id FROM Clients")
    validator.validate("SELECT Clients.Id FROM Clients")
    validator.validate("SELECT Clients.Id FROM Clients WHERE Clients.Id = 1")
    validator.validate(
        "SELECT Orders.Id FROM Orders JOIN Clients ON Clients.Id = Orders.ClientId"
    )

    with pytest.raises(QueryValidationError):
        validator.validate("SELECT Name FROM Clients")

    with pytest.raises(QueryValidationError):
        validator.validate("SELECT Clients.Name FROM Clients")

    with pytest.raises(QueryValidationError):
        validator.validate("SELECT Clients.Id FROM Clients WHERE Clients.Name = 'A'")

    with pytest.raises(QueryValidationError):
        validator.validate(
            "SELECT Orders.Id FROM Orders JOIN Clients ON Clients.Name = Orders.ClientName"
        )


def test_repository_blocks_clients_non_id_results(monkeypatch):
    repo = DatabaseRepository(
        connection_string=(
            "Driver={ODBC Driver 17 for SQL Server};Server=.;Database=Fake;Trusted_Connection=yes;"
        )
    )

    class _FakeCursor:
        description = [("Id",), ("Name",)]

        def execute(self, query):
            return None

        def fetchall(self):
            return [(1, "Alice")]

        def close(self):
            return None

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def close(self):
            return None

    @contextmanager
    def _fake_connection():
        yield _FakeConn()

    monkeypatch.setattr(repo, "_get_connection", _fake_connection)

    with pytest.raises(QueryValidationError):
        repo.execute_query("SELECT Clients.Name FROM Clients", validate=False)
