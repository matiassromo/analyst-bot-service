"""
Chart generation service using matplotlib and seaborn.
Creates visualizations from query results based on configurations.
"""

import base64
import io
import logging
from typing import List, Dict, Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

from app.models.llm_models import ChartConfig

logger = logging.getLogger(__name__)


class ChartGenerationError(Exception):
    """Raised when chart generation fails."""
    pass


class ChartService:
    """
    Service for generating charts from data.
    Uses Factory pattern to create different chart types.
    """

    # Default chart styling
    DEFAULT_FIGURE_SIZE = (10, 6)
    DEFAULT_DPI = 100
    DEFAULT_STYLE = 'whitegrid'

    def __init__(self):
        """Initialize chart service with default styling."""
        # Set seaborn style
        sns.set_style(self.DEFAULT_STYLE)
        sns.set_palette("husl")

        # Set default matplotlib parameters
        plt.rcParams['figure.figsize'] = self.DEFAULT_FIGURE_SIZE
        plt.rcParams['figure.dpi'] = self.DEFAULT_DPI
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 11

        logger.info("Chart service initialized")

    def generate_chart(
        self,
        data: List[Dict[str, Any]],
        config: ChartConfig
    ) -> str:
        """
        Generate a chart and return as base64-encoded PNG.

        Args:
            data: Query results as list of dictionaries
            config: Chart configuration

        Returns:
            Base64-encoded PNG image string

        Raises:
            ChartGenerationError: If chart generation fails
        """
        if not data:
            raise ChartGenerationError("Cannot generate chart: no data provided")

        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)

            # Validate columns exist
            self._validate_columns(df, config)

            # Generate chart based on type
            logger.info(f"Generating {config.type} chart: {config.title}")

            fig, ax = plt.subplots(figsize=self.DEFAULT_FIGURE_SIZE, dpi=self.DEFAULT_DPI)

            chart_generators = {
                "bar": self._create_bar_chart,
                "line": self._create_line_chart,
                "pie": self._create_pie_chart,
                "scatter": self._create_scatter_chart,
                "heatmap": self._create_heatmap,
            }

            generator = chart_generators.get(config.type)
            if not generator:
                raise ChartGenerationError(f"Unsupported chart type: {config.type}")

            generator(df, config, ax)

            # Convert to base64
            image_base64 = self._figure_to_base64(fig)

            # Clean up
            plt.close(fig)

            logger.info(f"Chart generated successfully ({len(image_base64)} bytes)")
            return image_base64

        except ChartGenerationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
            raise ChartGenerationError(f"Chart generation failed: {e}")

    def _validate_columns(self, df: pd.DataFrame, config: ChartConfig) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df: Data DataFrame
            config: Chart configuration

        Raises:
            ChartGenerationError: If required columns are missing
        """
        required_columns = [config.x_column]
        if config.y_column:
            required_columns.append(config.y_column)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ChartGenerationError(
                f"Required columns missing from data: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    def _create_bar_chart(
        self,
        df: pd.DataFrame,
        config: ChartConfig,
        ax: plt.Axes
    ) -> None:
        """Create a bar chart."""
        # Use seaborn for better styling
        sns.barplot(
            data=df,
            x=config.x_column,
            y=config.y_column,
            palette=config.color_palette,
            ax=ax
        )

        # Set labels and title
        ax.set_xlabel(config.x_label or config.x_column)
        ax.set_ylabel(config.y_label or config.y_column)
        ax.set_title(config.title, fontsize=14, fontweight='bold')

        # Rotate x labels if needed
        if len(df) > 10:
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')

    def _create_line_chart(
        self,
        df: pd.DataFrame,
        config: ChartConfig,
        ax: plt.Axes
    ) -> None:
        """Create a line chart."""
        sns.lineplot(
            data=df,
            x=config.x_column,
            y=config.y_column,
            marker='o',
            linewidth=2,
            markersize=6,
            ax=ax
        )

        ax.set_xlabel(config.x_label or config.x_column)
        ax.set_ylabel(config.y_label or config.y_column)
        ax.set_title(config.title, fontsize=14, fontweight='bold')

        # Rotate x labels if many data points
        if len(df) > 15:
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

        ax.grid(True, alpha=0.3)

    def _create_pie_chart(
        self,
        df: pd.DataFrame,
        config: ChartConfig,
        ax: plt.Axes
    ) -> None:
        """Create a pie chart."""
        # Limit to top 10 slices for readability
        if len(df) > 10:
            logger.warning(f"Pie chart has {len(df)} slices, limiting to top 10")
            # Find numeric column for sorting (nlargest requires numeric dtype)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                df = df.nlargest(10, numeric_cols[0])
            else:
                # Fallback: just take first 10 rows if no numeric column found
                df = df.head(10)

        # Create pie chart
        colors = sns.color_palette(config.color_palette, len(df))

        wedges, texts, autotexts = ax.pie(
            df[config.x_column],
            labels=df.index if df.index.name else df[df.columns[0]],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(config.title, fontsize=14, fontweight='bold')

    def _create_scatter_chart(
        self,
        df: pd.DataFrame,
        config: ChartConfig,
        ax: plt.Axes
    ) -> None:
        """Create a scatter plot."""
        sns.scatterplot(
            data=df,
            x=config.x_column,
            y=config.y_column,
            s=100,
            alpha=0.6,
            color=sns.color_palette(config.color_palette)[0],
            ax=ax
        )

        ax.set_xlabel(config.x_label or config.x_column)
        ax.set_ylabel(config.y_label or config.y_column)
        ax.set_title(config.title, fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3)

    def _create_heatmap(
        self,
        df: pd.DataFrame,
        config: ChartConfig,
        ax: plt.Axes
    ) -> None:
        """Create a heatmap."""
        # For heatmap, we need to pivot the data or use it as-is if already in matrix form
        # Assume the data is in long format and needs pivoting
        if config.y_column and len(df.columns) >= 3:
            # Pivot data for heatmap
            pivot_table = df.pivot_table(
                index=config.x_column,
                columns=config.y_column,
                values=df.columns[2],  # Use third column as values
                aggfunc='sum'
            )
        else:
            # Use data as-is (assume it's already in matrix form)
            pivot_table = df.set_index(config.x_column)

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap=config.color_palette,
            cbar_kws={'label': config.y_label or 'Value'},
            ax=ax
        )

        ax.set_xlabel(config.x_label or config.x_column)
        ax.set_ylabel(config.y_label or config.y_column)
        ax.set_title(config.title, fontsize=14, fontweight='bold')

        plt.tight_layout()

    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG.

        Args:
            fig: Matplotlib figure

        Returns:
            Base64-encoded PNG string
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=self.DEFAULT_DPI)
        buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        return image_base64

    def generate_multiple_charts(
        self,
        data: List[Dict[str, Any]],
        configs: List[ChartConfig]
    ) -> List[Dict[str, str]]:
        """
        Generate multiple charts from the same data.

        Args:
            data: Query results
            configs: List of chart configurations

        Returns:
            List of dictionaries with chart metadata and base64 images
        """
        charts = []

        for config in configs:
            try:
                image_base64 = self.generate_chart(data, config)
                charts.append({
                    "type": config.type,
                    "title": config.title,
                    "image_base64": image_base64
                })
            except ChartGenerationError as e:
                logger.error(f"Failed to generate {config.type} chart '{config.title}': {e}")
                # Continue with other charts
                continue

        return charts

    @staticmethod
    def decode_base64_image(base64_string: str) -> Image.Image:
        """
        Decode base64 string to PIL Image.
        Utility method for testing or further processing.

        Args:
            base64_string: Base64-encoded image string

        Returns:
            PIL Image object
        """
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
