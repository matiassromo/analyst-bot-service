"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables with type validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


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
    allowed_ips: str = "127.0.0.1"

    # Database Configuration
    db_server: str
    db_port: int = 1433
    db_name: str
    db_user: str
    db_password: str
    db_driver: str = "ODBC Driver 17 for SQL Server"

    # Google Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-pro"

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    max_query_rows: int = 10000
    query_timeout_seconds: int = 30

    @property
    def allowed_ips_list(self) -> List[str]:
        """
        Parse comma-separated IPs into a list.
        Supports individual IPs and CIDR notation.

        Examples:
            - "127.0.0.1" -> ["127.0.0.1"]
            - "192.168.1.1,10.0.0.0/24" -> ["192.168.1.1", "10.0.0.0/24"]
        """
        return [ip.strip() for ip in self.allowed_ips.split(",") if ip.strip()]

    @property
    def db_connection_string(self) -> str:
        """
        Generate SQL Server connection string for pyodbc.

        Format: DRIVER={driver};SERVER=host,port;DATABASE=db;UID=user;PWD=pass
        """
        return (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_server},{self.db_port};"
            f"DATABASE={self.db_name};"
            f"UID={self.db_user};"
            f"PWD={self.db_password}"
        )

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
