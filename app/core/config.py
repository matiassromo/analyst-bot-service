"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables with type validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via .env file or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "Analyst Bot Service"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Database Configuration
    db_server: str
    db_port: int = 1433
    db_name: str
    db_user: str = ""
    db_password: str = ""
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_auth_type: Literal["windows", "sql", "auto"] = "auto"
    db_trust_server_certificate: bool = False

    # Google Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # LLM Global Context
    llm_global_context: str = ""
    llm_global_context_path: str = ""

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    max_query_rows: int = 10000
    query_timeout_seconds: int = 30

    @property
    def is_azure_sql(self) -> bool:
        """Check if the server is an Azure SQL Database."""
        return ".database.windows.net" in self.db_server.lower()

    @property
    def db_connection_string(self) -> str:
        """
        Generate SQL Server connection string for pyodbc.

        Supports two authentication modes:
        - Windows Auth: Trusted_Connection=yes (local SQL Server)
        - SQL Auth: UID/PWD with encryption (Azure SQL or explicit SQL auth)

        The auth mode is determined by db_auth_type:
        - "windows": Always use Windows Authentication
        - "sql": Always use SQL Authentication
        - "auto": Use SQL Auth for Azure (*.database.windows.net), Windows Auth otherwise
        """
        base = (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_server};"
            f"DATABASE={self.db_name};"
        )

        # Determine effective auth type
        if self.db_auth_type == "auto":
            use_sql_auth = self.is_azure_sql
        else:
            use_sql_auth = self.db_auth_type == "sql"

        if use_sql_auth:
            return (
                base +
                f"UID={self.db_user};"
                f"PWD={self.db_password};"
                f"Encrypt=yes;"
                f"TrustServerCertificate=no;"
            )
        else:
            trust = "yes" if self.db_trust_server_certificate else "no"
            return base + f"Trusted_Connection=yes;TrustServerCertificate={trust};"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


# Global settings instance
settings = Settings()
