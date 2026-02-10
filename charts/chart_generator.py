import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime, timedelta
import os
import sys
from typing import Optional, List, Dict, Any, Tuple

# Add parent directory to path to import data_connector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_connector import DataConnector
from metric_calculator import MetricCalculator

class ChartGenerator:
    """Dynamic chart generator based on NER extracted entities - NO HARDCODING"""
    
    # Calculated metric columns to filter out when detecting data structure
    CALCULATED_COLUMNS = ['YoY_Growth_Pct', 'MoM_Change_Pct', 'Cumulative_Total', 'Previous_Value']
    
    def __init__(self, data_sources=None, auto_load_data=True, data_folder=None):
        """
        Initialize chart generator with real data sources only
        data_sources: dict of dataframes or data connections
        auto_load_data: if True, automatically loads all data files; if False, loads only uploaded files
        data_folder: absolute path to data folder (if None, uses default relative path)
        """
        self.data_sources = data_sources or {}
        
        # Enhanced color palettes for beautiful charts
        self.color_palettes = {
            'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#F2CC8F'],
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
            'professional': ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E'],
            'modern': ['#667eea', '#f093fb', '#4facfe', '#00f2fe', '#fa709a', '#fee140', '#a8edea'],
            'corporate': ['#1f4e79', '#2d5aa0', '#5b9bd5', '#70ad47', '#ffc000', '#c5504b', '#7030a0']
        }
        self.default_colors = self.color_palettes['business']
        
        # Initialize data connector for real data - MANDATORY
        try:
            # Use provided data_folder or default to relative path
            if data_folder is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                data_folder = os.path.join(os.path.dirname(script_dir), 'data')
            
            self.data_connector = DataConnector(data_folder=data_folder, auto_load=auto_load_data)
            if auto_load_data:
                print("📊 Data connector initialized with real data sources")
            else:
                print("📊 Data connector initialized - waiting for data upload")
        except Exception as e:
            raise Exception(f"❌ Could not initialize data connector: {e}. System requires real data sources.")
        
        # Initialize metric calculator for calculated metrics (YoY growth, etc.)
        self.metric_calculator = MetricCalculator()
        print("🧮 Metric calculator initialized for calculated metrics")
        
        # Display available data info for transparency (only if auto_load_data is True)
        if auto_load_data:
            self.display_available_data()
        
    def display_available_data(self):
        """Display information about available data sources"""
        if self.data_connector:
            print("\n🗂️ AVAILABLE DATA SOURCES:")
            print("=" * 60)
            columns_info = self.data_connector.extract_all_columns_info()
            
            for table_name, info in columns_info.items():
                print(f"\n📊 Table: {table_name.upper()}")
                print(f"   📈 Rows: {info['row_count']:,}")
                print(f"   📋 All Columns: {info['all_columns']}")
                print(f"   🔢 Numeric Columns: {info['numeric_columns']}")
                print(f"   📝 Text Columns: {info['text_columns']}")
                if info['date_columns']:
                    print(f"   📅 Date Columns: {info['date_columns']}")
            print("=" * 60)
            print("💡 System will automatically match query entities to these columns\n")
        
    def add_data_source(self, name, dataframe):
        """Add a data source (DataFrame) to the generator"""
        self.data_sources[name] = dataframe
    
    def _get_base_columns(self, data: pd.DataFrame) -> List[str]:
        """Filter out calculated columns to get base data structure"""
        return [col for col in data.columns 
                if col not in self.CALCULATED_COLUMNS 
                and not col.startswith('MA_')]
    
    def _detect_multi_series(self, data: pd.DataFrame) -> bool:
        """Detect if data has multi-series structure (dimension, group_by, metric)"""
        base_columns = self._get_base_columns(data)
        has_group_by = any('group_by_' in col for col in base_columns)
        return len(base_columns) == 3 and has_group_by
    
    def _detect_duplicate_dimensions(self, data: pd.DataFrame, dimension_col: str) -> bool:
        """Check if dimension column has duplicate values (indicating multi-entity data)"""
        return data[dimension_col].duplicated().any()
    
    def _infer_grouping_column(self, data: pd.DataFrame, dimension_col: str) -> Optional[str]:
        """
        Intelligently infer which column is the grouping/category column when
        we detect multiple values per dimension (e.g., multiple states per fiscal year).
        
        Returns the column name that likely represents the series grouping,
        or None if ambiguous.
        """
        base_columns = self._get_base_columns(data)
        
        # If only 2 columns, can't be multi-series
        if len(base_columns) < 3:
            return None
        
        # Column is likely a grouping column if:
        # 1. It's not the dimension column (first column)
        # 2. It's not numeric (not the metric)
        # 3. It has reasonable cardinality (2-50 unique values)
        
        candidate_cols = [col for col in base_columns if col != dimension_col]
        
        for col in candidate_cols:
            # Skip if it's numeric (likely the metric)
            if pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            # Check cardinality
            unique_count = data[col].nunique()
            if 2 <= unique_count <= 50:  # Reasonable number of series
                return col
        
        return None
    
    def _prepare_data_for_chart(self, data: pd.DataFrame, target_metric: str, 
                                is_multi_series: bool, base_columns: List[str]) -> pd.DataFrame:
        """
        Prepare data for chart by selecting only the relevant columns.
        Handles calculated metrics (Cumulative_Total, YoY_Growth_Pct, etc.)
        
        Args:
            data: Full dataframe with all columns including calculated ones
            target_metric: The metric to chart (could be a calculated column or base column)
            is_multi_series: Whether this is multi-series data
            base_columns: List of base columns (excluding calculated ones)
        
        Returns:
            DataFrame with only the columns needed for charting
        """
        # Check if target_metric is a calculated column
        calculated_cols = ['YoY_Growth_Pct', 'MoM_Change_Pct', 'Cumulative_Total', 'Previous_Value']
        calculated_cols.extend([col for col in data.columns if col.startswith('MA_')])
        
        is_calculated = target_metric in calculated_cols or target_metric in data.columns
        
        if is_multi_series:
            # Multi-series: dimension, group_by, metric
            dimension_col = base_columns[0]
            group_by_col = base_columns[1]
            
            # If target_metric is a calculated column, use it; otherwise use base metric
            if is_calculated and target_metric in data.columns:
                metric_col = target_metric
            else:
                metric_col = base_columns[2]
            
            result = data[[dimension_col, group_by_col, metric_col]].copy()
            print(f"🔧 Prepared multi-series data: {dimension_col}, {group_by_col}, {metric_col}")
        else:
            # Single-series: dimension, metric
            if len(base_columns) >= 2:
                dimension_col = base_columns[0]
                
                # If target_metric is a calculated column, use it; otherwise use base metric
                if is_calculated and target_metric in data.columns:
                    metric_col = target_metric
                else:
                    metric_col = base_columns[1]
                
                result = data[[dimension_col, metric_col]].copy()
                print(f"🔧 Prepared single-series data: {dimension_col}, {metric_col}")
            else:
                # Edge case: only one column
                result = data.copy()
        
        return result
    
    def _get_calculated_column(self, data: pd.DataFrame) -> Optional[str]:
        """Get the calculated metric column if present"""
        calculated_cols = [col for col in data.columns
                          if col in self.CALCULATED_COLUMNS or col.startswith('MA_')]
        return calculated_cols[0] if calculated_cols else None
    
    def _get_smart_yaxis_range(self, values: pd.Series, metric_name: str) -> Optional[Tuple[float, float]]:
        """
        Calculate smart Y-axis range for metrics with potential outliers.
        
        Uses percentile-based approach to focus on the meaningful data range
        while indicating presence of outliers.
        
        Args:
            values: Pandas Series of metric values
            metric_name: Name of the metric (for logging)
            
        Returns:
            (min, max) tuple for Y-axis range, or None if no adjustment needed
        """
        # Remove NaN values
        clean_values = values.dropna()
        
        if len(clean_values) == 0:
            return None
            
        # Calculate percentiles for robust outlier detection
        p5 = clean_values.quantile(0.05)  # 5th percentile
        p95 = clean_values.quantile(0.95)  # 95th percentile
        median = clean_values.median()
        
        # Calculate IQR for outlier detection
        q1 = clean_values.quantile(0.25)
        q3 = clean_values.quantile(0.75)
        iqr = q3 - q1
        
        # Identify extreme outliers (beyond 3x IQR)
        lower_fence = q1 - 3 * iqr
        upper_fence = q3 + 3 * iqr
        outliers = clean_values[(clean_values < lower_fence) | (clean_values > upper_fence)]
        
        # If there are significant outliers (>5% of data or extreme values)
        if len(outliers) > len(clean_values) * 0.05 or (p95 - p5) > abs(median) * 10:
            # Use 5th-95th percentile range with 10% padding
            range_span = p95 - p5
            padding = range_span * 0.1
            
            y_min = p5 - padding
            y_max = p95 + padding
            
            print(f"⚠️  Outlier protection: {len(outliers)} extreme values detected in {metric_name}")
            print(f"   Setting Y-axis range to [{y_min:.1f}, {y_max:.1f}] (5th-95th percentile ±10%)")
            print(f"   Outlier range: [{clean_values.min():.1f}, {clean_values.max():.1f}]")
            
            return (y_min, y_max)
        
        # No significant outliers - let Plotly auto-scale
        return None

    def _get_extended_color_palette(self, num_colors: int, palette_name: str = 'professional') -> List[str]:
        """
        Generate extended color palette with HIGHLY DISTINCT colors for large number of categories.
        Uses HSV color wheel distribution for maximum visual separation.

        Args:
            num_colors: Number of unique colors needed
            palette_name: Base palette to extend (used for small numbers)

        Returns:
            List of color hex codes
        """
        base_colors = self.color_palettes.get(palette_name, self.default_colors)

        # For small numbers, use curated base palette
        if num_colors <= len(base_colors):
            return base_colors[:num_colors]

        # For larger numbers, generate maximally distinct colors using HSV color wheel
        import colorsys
        extended_palette = []
        
        # Strategy: Distribute colors evenly across hue spectrum with varying saturation/value
        # This creates a "rainbow effect" with maximum visual distinction
        
        # Calculate how many "rounds" we need through the color wheel
        colors_per_round = 12  # Distribute 12 distinct hues per round
        num_rounds = (num_colors + colors_per_round - 1) // colors_per_round
        
        color_index = 0
        for round_num in range(num_rounds):
            # Alternate saturation and value for each round to create variety
            if round_num == 0:
                # First round: Full saturation, high value (vibrant colors)
                s_base, v_base = 0.85, 0.90
            elif round_num == 1:
                # Second round: Medium saturation, full value (lighter colors)
                s_base, v_base = 0.60, 0.95
            elif round_num == 2:
                # Third round: Full saturation, medium value (darker colors)
                s_base, v_base = 0.90, 0.65
            else:
                # Additional rounds: Vary systematically
                s_base = 0.7 + (round_num % 3) * 0.1
                v_base = 0.7 + ((round_num + 1) % 3) * 0.1
            
            # Generate colors evenly distributed around the hue wheel
            for i in range(colors_per_round):
                if color_index >= num_colors:
                    break
                    
                # Distribute hues evenly (0.0 to 1.0)
                # Offset each round slightly for better distribution
                hue = (i / colors_per_round + round_num * 0.04) % 1.0
                
                # Add slight variation to saturation and value for more distinction
                s = max(0.4, min(1.0, s_base + (i % 3 - 1) * 0.05))
                v = max(0.5, min(1.0, v_base + (i % 2) * 0.05))
                
                # Convert HSV to RGB
                r, g, b = colorsys.hsv_to_rgb(hue, s, v)
                hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                extended_palette.append(hex_color)
                color_index += 1
        
        print(f"🎨 Generated {len(extended_palette)} distinct colors for visualization")
        return extended_palette

    def _apply_standard_style(self, fig, title: str, x_title: str, y_title: str,
                              show_legend: bool = False, legend_title: str = None,
                              yaxis_range: Tuple[float, float] = None) -> go.Figure:
        """
        Apply standard styling to chart figure (DRY principle).
        Centralizes all layout configurations to avoid repetition.

        Args:
            fig: Plotly figure object
            title: Chart title
            x_title: X-axis label
            y_title: Y-axis label
            show_legend: Whether to show legend
            legend_title: Optional legend title
            yaxis_range: Optional (min, max) tuple to set Y-axis range

        Returns:
            Styled figure
        """
        layout_config = {
            'title': {
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            'xaxis_title': x_title,
            'yaxis_title': y_title,
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'family': 'Arial', 'color': '#34495E'},
            'height': 500,
            'margin': dict(t=80, l=60, r=60, b=60),
            'template': 'plotly_white',
            'showlegend': show_legend
        }
        
        fig.update_layout(**layout_config)

        # Add legend positioning if needed
        if show_legend:
            fig.update_layout(
                legend=dict(
                    title={'text': legend_title} if legend_title else None,
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                )
            )

        # Standard grid styling
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECF0F1', linecolor='#BDC3C7', linewidth=2)
        fig.update_xaxes(showgrid=False, linecolor='#BDC3C7', linewidth=2)
        
        # Apply Y-axis range if specified (for handling outliers)
        if yaxis_range is not None:
            fig.update_yaxes(range=yaxis_range)

        return fig

    def _sort_fiscal_year_data(self, data: pd.DataFrame, dimension_col: str) -> pd.DataFrame:
        """
        Detect and sort fiscal year and month data chronologically.
        Handles formats:
        - FY: 2020-21, 2019-20, FY2020-21, FY 2020-21, 2020-2021
        - Months: Jan, January, Feb, February, etc.
        - Month-Year: Jan 2020, January 2020, Jan-20, etc.

        Args:
            data: DataFrame with potential FY/month data
            dimension_col: Column name to check and sort

        Returns:
            Sorted DataFrame (or original if not FY/month data)
        """
        if dimension_col not in data.columns:
            return data

        # Sample first few values to detect pattern
        sample_values = data[dimension_col].astype(str).head(10).tolist()

        import re

        # Regex patterns for fiscal year formats
        fy_patterns = [
            r'^\d{4}-\d{2}$',          # 2020-21
            r'^FY\s*\d{4}-\d{2}$',     # FY2020-21, FY 2020-21
            r'^\d{4}-\d{4}$',          # 2020-2021
            r'^FY\s*\d{4}-\d{4}$',     # FY2020-2021
        ]

        # Check if FY pattern
        is_fy = False
        for pattern in fy_patterns:
            if any(re.match(pattern, str(val).strip(), re.IGNORECASE) for val in sample_values if val):
                is_fy = True
                break

        # Month name mapping for sorting
        month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

        # Check if month pattern
        is_month = False
        if not is_fy:
            # Check if any sample value contains month names
            for val in sample_values:
                if val and any(month.lower() in str(val).lower() for month in month_map.keys()):
                    is_month = True
                    break

        if not is_fy and not is_month:
            return data

        if is_fy:
            print(f"🔍 Detected fiscal year format in '{dimension_col}' - applying chronological sort")

            # Extract starting year from FY string for sorting
            def extract_fy_year(fy_str):
                """Extract the starting year from FY string for sorting"""
                try:
                    fy_str = str(fy_str).strip()
                    # Remove 'FY' prefix if present
                    fy_str = re.sub(r'^FY\s*', '', fy_str, flags=re.IGNORECASE)
                    # Extract first 4 digits
                    match = re.search(r'\d{4}', fy_str)
                    if match:
                        return int(match.group())
                    return 9999  # Put unparseable values at the end
                except:
                    return 9999

            # Create temporary sort key
            data_copy = data.copy()
            data_copy['_fy_sort_key'] = data_copy[dimension_col].apply(extract_fy_year)
            data_sorted = data_copy.sort_values('_fy_sort_key').drop('_fy_sort_key', axis=1)

            sorted_values = data_sorted[dimension_col].unique()[:5]
            print(f"   ✅ Sorted FY data: {', '.join(map(str, sorted_values))}...")

        elif is_month:
            print(f"🔍 Detected month format in '{dimension_col}' - applying chronological sort")

            def extract_month_sort_key(month_str):
                """Extract month and year for sorting"""
                try:
                    month_str = str(month_str).strip().lower()

                    # Try to find month name
                    month_num = None
                    for month_name, num in month_map.items():
                        if month_name in month_str:
                            month_num = num
                            break

                    if month_num is None:
                        return 999999  # Put unparseable at end

                    # Try to extract year (4 digits or 2 digits)
                    year_match = re.search(r'(\d{4})|(\d{2})(?!\d)', month_str)
                    if year_match:
                        year_str = year_match.group(1) or year_match.group(2)
                        year = int(year_str)
                        # Convert 2-digit year to 4-digit (assume 2000s for 00-50, 1900s for 51-99)
                        if year < 100:
                            year = 2000 + year if year <= 50 else 1900 + year
                        # Sort key: YYYYMM format
                        return year * 100 + month_num
                    else:
                        # No year found, just use month (for data like "Jan", "Feb", etc.)
                        return month_num
                except:
                    return 999999

            # Create temporary sort key
            data_copy = data.copy()
            data_copy['_month_sort_key'] = data_copy[dimension_col].apply(extract_month_sort_key)
            data_sorted = data_copy.sort_values('_month_sort_key').drop('_month_sort_key', axis=1)

            sorted_values = data_sorted[dimension_col].unique()[:5]
            print(f"   ✅ Sorted month data: {', '.join(map(str, sorted_values))}...")

        return data_sorted

    def get_chart_data(
        self, 
        metric: Optional[str], 
        dimension: Optional[str], 
        filters: Optional[List[str]] = None, 
        aggregation: Optional[str] = None, 
        query_text: str = "", 
        group_by: Optional[str] = None,
        calculation_type: Optional[str] = None,
        calculation_window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get data for chart generation - COLUMN-BASED APPROACH
        Priority: Actual column names in query > NER entities
        
        Args:
            metric: Metric column name
            dimension: Dimension column name
            filters: List of filter conditions
            aggregation: Aggregation function (sum, avg, count, etc.)
            query_text: Original natural language query
            group_by: Optional secondary dimension for multi-series charts (creates multiple lines/colors)
            calculation_type: Type of calculation (yoy_growth, mom_change, cumulative, etc.)
            calculation_window: Window size for moving average calculations
            
        Returns:
            DataFrame with chart data, optionally with calculated metrics
        """
        if not self.data_connector:
            raise Exception("❌ No data connector available. Cannot generate charts without real data.")
            
        try:
            # Use the new column-based approach with group_by support
            data = self.data_connector.get_data_for_chart_column_based(
                metric, dimension, filters, aggregation, query_text, group_by
            )
            if data is None or data.empty:
                raise Exception(f"❌ No data found for query: {query_text}")
            
            # Apply calculated metrics if requested
            if calculation_type:
                print(f"🧮 Applying calculation: {calculation_type}")
                
                # Identify metric and dimension columns from the returned DataFrame
                # Format: dimension_<name>, metric_<name>, or group_by_<name>
                metric_col = None
                dimension_col = None
                
                for col in data.columns:
                    if col.startswith('metric_'):
                        metric_col = col
                    elif col.startswith('dimension_'):
                        dimension_col = col
                
                if metric_col and dimension_col:
                    data = self.metric_calculator.calculate_metric(
                        data, calculation_type, metric_col, dimension_col, calculation_window
                    )
                    print(f"✅ Calculated metric applied: {calculation_type}")
                else:
                    print(f"⚠️ Could not identify metric/dimension columns for calculation")
            
            return data
        except Exception as e:
            print(f"❌ Error getting real data: {e}")
            raise Exception(f"❌ Failed to retrieve data for chart generation: {e}")
    
    def auto_detect_chart_type(self, metric, dimension, data):
        """
        Automatically detect the best chart type based on data characteristics
        """
        if data is None or data.empty:
            return 'table'
            
        num_categories = len(data)
        dimension_col = data.columns[0]
        unique_dims = data[dimension_col].nunique()
        
        # Check if dimension appears to be time-based
        dimension_values = data.iloc[:, 0].astype(str)
        is_time_based = any(
            pattern in str(dimension).lower() 
            for pattern in ['date', 'time', 'month', 'quarter', 'year', 'day']
        ) or any(
            re.search(r'\d{4}', str(val)) or 
            any(month in str(val).lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun'])
            for val in dimension_values
        )
        
        # Chart type selection logic
        if unique_dims == 1:
            print(f"📊 Dimension has only 1 unique value - suggesting card/KPI view")
            return 'card'
        elif is_time_based:
            return 'line'
        elif unique_dims <= 8 and 'distribution' in str(metric).lower():
            return 'pie'
        elif unique_dims <= 8:
            return 'pie'
        elif unique_dims <= 50:
            return 'bar'
        elif unique_dims <= 100:
            return 'table'
        else:
            return 'table'  # Default fallback
    
    def create_bar_chart(self, data, metric, dimension, title=None):
        """Create a beautiful bar chart from real data with enhanced colors (supports grouped bars)"""
        if data is None or data.empty:
            raise Exception("❌ No data available for bar chart generation")

        # Reset index to ensure we're not plotting index values
        data = data.reset_index(drop=True)

        # Check if this is multi-series data (3 columns: dimension, group_by, metric)
        is_multi_series = len(data.columns) == 3 and 'group_by_' in data.columns[1]

        if is_multi_series:
            # Multi-series grouped bar chart - THE PLOTLY EXPRESS WAY
            dimension_col = data.columns[0]  # First column is dimension (X-axis)
            group_by_col = data.columns[1]   # Second column is group_by (color groups)
            metric_col = data.columns[2]     # Third column is metric (Y-axis)

            # Sort fiscal year data if detected
            data = self._sort_fiscal_year_data(data, dimension_col)

            # Ensure numeric column is actually numeric
            data[metric_col] = pd.to_numeric(data[metric_col], errors='coerce')

            num_categories = data[group_by_col].nunique()
            print(f"📊 Multi-series grouped bar: {num_categories} categories × {len(data[dimension_col].unique())} dimensions")

            # Get extended color palette to avoid repetition
            colors = self._get_extended_color_palette(num_categories, 'professional')

            # Use Plotly Express - automatic grouping by color parameter
            fig = px.bar(
                data,
                x=dimension_col,
                y=metric_col,
                color=group_by_col,  # This does the loop for you!
                barmode='group',
                color_discrete_sequence=colors,
                title=title or f"{metric.title()} by {dimension.title()} (grouped by {group_by_col.replace('group_by_', '')})"
            )

            # Apply standard styling
            legend_title = group_by_col.replace('group_by_', '').title()
            self._apply_standard_style(fig,
                                       title or f"{metric.title()} by {dimension.title()} (grouped by {legend_title})",
                                       dimension.title(),
                                       metric.title(),
                                       show_legend=True,
                                       legend_title=legend_title)

            # Adjust bottom margin for angled labels
            fig.update_layout(margin=dict(b=120), xaxis_tickangle=-45)

            # Add white borders to bars
            fig.update_traces(marker_line_color='white', marker_line_width=1)

            print(f"   ✅ Created grouped bar chart with {num_categories} groups using Plotly Express")

            return fig

        # Single-series bar chart (original logic)
        metric_col = data.columns[1]  # Second column is metric
        dimension_col = data.columns[0]  # First column is dimension

        # Sort fiscal year data if detected
        data = self._sort_fiscal_year_data(data, dimension_col)

        # Data is already aggregated in data_connector, so use it directly
        aggregated_data = data.copy()

        # Debugging: print data snapshot and dtypes to diagnose empty plots
        try:
            print("🔎 create_bar_chart - snapshot of aggregated_data:")
            print(aggregated_data.head(10).to_dict(orient='records'))
            print("🔎 dtypes:", aggregated_data.dtypes.to_dict())
        except Exception as _:
            print("🔎 create_bar_chart - could not print data snapshot")

        # Ensure metric column is numeric (coerce common formatting)
        try:
            aggregated_data[metric_col] = pd.to_numeric(
                aggregated_data[metric_col].astype(str).str.replace(',', '').str.replace('$', ''),
                errors='coerce'
            )
            if aggregated_data[metric_col].isna().all():
                print(f"⚠️ create_bar_chart: metric column '{metric_col}' contains no numeric values after coercion")
        except Exception as e:
            print(f"⚠️ create_bar_chart: error coercing metric to numeric: {e}")

        # Ensure dimension column is string-like for plotting
        try:
            aggregated_data[dimension_col] = aggregated_data[dimension_col].astype(str)
        except Exception as e:
            print(f"⚠️ create_bar_chart: error casting dimension to string: {e}")

        # Check dimension cardinality
        unique_dims = len(aggregated_data)

        if unique_dims == 1:
            print(f"⚠️  Warning: Dimension '{dimension}' has only 1 unique value. Bar chart will show only one bar.")
            print(f"   Consider using a different dimension or chart type.")
        elif unique_dims > 50:
            print(f"⚠️  Warning: Dimension '{dimension}' has {unique_dims} unique values. Limiting to top 20 for better visualization.")
            # Keep only top 20 categories (data is already aggregated)
            aggregated_data = aggregated_data.nlargest(20, metric_col).reset_index(drop=True)
            unique_dims = len(aggregated_data)

        # Get extended color palette to avoid repetition
        colors = self._get_extended_color_palette(unique_dims, 'professional')
        
        # Verify we have enough colors
        if len(colors) < unique_dims:
            print(f"⚠️  Color mismatch: {len(colors)} colors for {unique_dims} bars - extending...")
            colors = colors * ((unique_dims // len(colors)) + 1)
            colors = colors[:unique_dims]

        # Extract values explicitly to avoid any Plotly confusion
        x_values = aggregated_data[dimension_col].tolist()
        y_values = aggregated_data[metric_col].tolist()

        print(f"   📊 Plotting {unique_dims} bars with {len(colors)} unique colors")
        print(f"   📊 X values: {x_values[:5]}{'...' if len(x_values) > 5 else ''}")
        print(f"   📊 Y values: {y_values[:5]}{'...' if len(y_values) > 5 else ''}")

        # Create figure using graph_objects for more control
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker=dict(
                color=colors[:unique_dims],  # Ensure exact color count
                line=dict(color='white', width=1.5)  # Thicker white borders for separation
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         f'{metric.title()}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        ))

        # Apply standard styling
        self._apply_standard_style(fig,
                                   title or f"{metric.title()} by {dimension.title()}",
                                   dimension.title(),
                                   metric.title(),
                                   show_legend=False)

        # Adjust for angled labels if many categories
        if unique_dims > 8:
            fig.update_layout(margin=dict(b=120), xaxis_tickangle=-45)

        return fig
    
    def create_line_chart(self, data, metric, dimension, title=None):
        """Create a beautiful line chart from real data with enhanced colors (supports multi-series)"""
        if data is None or data.empty:
            raise Exception("❌ No data available for line chart generation")

        # Reset index to ensure we're not plotting index values
        data = data.reset_index(drop=True)

        # Detect multi-series structure using helper methods
        is_multi_series = self._detect_multi_series(data)
        base_columns = self._get_base_columns(data)
        calculated_col = self._get_calculated_column(data)
        
        # INTELLIGENT MULTI-SERIES DETECTION
        # If not explicitly multi-series, check if we have duplicate dimension values
        if not is_multi_series and len(base_columns) >= 2:
            dimension_col = base_columns[0]
            
            if self._detect_duplicate_dimensions(data, dimension_col):
                # Multiple values per dimension - need to handle as multi-series
                grouping_col = self._infer_grouping_column(data, dimension_col)
                
                if grouping_col:
                    print(f"⚠️  Detected multiple values per {dimension_col} - treating as multi-series grouped by {grouping_col}")
                    # Rearrange columns to match multi-series structure
                    metric_col = calculated_col if calculated_col else base_columns[-1]
                    data = data[[dimension_col, grouping_col, metric_col]].copy()
                    base_columns = [dimension_col, grouping_col, metric_col]
                    is_multi_series = True
                else:
                    # Can't determine grouping - aggregate instead
                    print(f"⚠️  Found {len(data)} values for {data[dimension_col].nunique()} time periods")
                    print(f"   Aggregating by averaging multiple values per period...")
                    metric_col = calculated_col if calculated_col else base_columns[-1]
                    data = data.groupby(dimension_col)[metric_col].mean().reset_index()
                    base_columns = [dimension_col, metric_col]
                    is_multi_series = False

        if is_multi_series:
            # Multi-series chart - THE PLOTLY EXPRESS WAY
            dimension_col = base_columns[0]
            group_by_col = base_columns[1]
            metric_col = calculated_col if calculated_col else base_columns[2]

            # Sort fiscal year data if detected (CRITICAL for line charts!)
            data = self._sort_fiscal_year_data(data, dimension_col)
            
            # Handle outliers in YoY/percentage metrics to prevent chart distortion
            yaxis_range = None
            is_calculated_metric = metric_col in ['YoY_Growth_Pct', 'MoM_Change_Pct'] or 'pct' in metric_col.lower() or 'growth' in metric_col.lower()
            if is_calculated_metric:
                yaxis_range = self._get_smart_yaxis_range(data[metric_col], metric_col)

            # Check for NaN values that would cause line gaps
            nan_count = data[metric_col].isna().sum()
            if nan_count > 0:
                print(f"⚠️  Found {nan_count} missing/unreliable values in {metric_col} - will connect across gaps")

            num_lines = data[group_by_col].nunique()
            print(f"📊 Multi-series line: {num_lines} lines over {len(data[dimension_col].unique())} periods")

            # Get extended color palette to avoid repetition
            colors = self._get_extended_color_palette(num_lines, 'professional')

            # Use Plotly Express - automatic line splitting by color parameter
            fig = px.line(
                data,
                x=dimension_col,
                y=metric_col,
                color=group_by_col,  # This does the loop for you!
                markers=True,
                color_discrete_sequence=colors
            )

            # Style the markers and lines, connect across NaN gaps for calculated metrics
            fig.update_traces(
                line=dict(width=2.5),
                marker=dict(size=6, line=dict(width=1, color='white')),
                connectgaps=True  # Connect across NaN values (from outlier protection or first period)
            )

            # Apply standard styling with smart Y-axis range
            legend_title = group_by_col.replace('group_by_', '').title()
            self._apply_standard_style(fig,
                                       title or f"{metric.title()} Trend by {dimension.title()}",
                                       dimension.title(),
                                       metric.title(),
                                       show_legend=True,
                                       legend_title=legend_title,
                                       yaxis_range=yaxis_range)

            print(f"   ✅ Created multi-series line chart with {num_lines} lines using Plotly Express")

        else:
            # Single-series: dimension + metric
            dimension_col = base_columns[0]
            metric_col = calculated_col if calculated_col else (base_columns[1] if len(base_columns) >= 2 else base_columns[0])

            # Sort fiscal year data if detected (CRITICAL for line charts!)
            data = self._sort_fiscal_year_data(data, dimension_col)
            
            # Handle outliers in YoY/percentage metrics to prevent chart distortion
            yaxis_range = None
            is_calculated_metric = metric_col in ['YoY_Growth_Pct', 'MoM_Change_Pct'] or 'pct' in metric_col.lower() or 'growth' in metric_col.lower()
            if is_calculated_metric:
                yaxis_range = self._get_smart_yaxis_range(data[metric_col], metric_col)

            # Check for NaN values that would cause line gaps
            nan_count = data[metric_col].isna().sum()
            if nan_count > 0:
                print(f"⚠️  Found {nan_count} missing/unreliable values in {metric_col} - will connect across gaps")

            # Get data values
            x_values = data[dimension_col].tolist()
            y_values = data[metric_col].tolist()

            import plotly.graph_objects as go
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=metric.title(),
                line=dict(width=3, color='#3498DB'),
                marker=dict(size=8, color='#E74C3C', line=dict(width=2, color='white')),
                connectgaps=True,  # Connect across NaN values (from outlier protection or first period)
                hovertemplate='<b>%{x}</b><br>' +
                             f'{metric.title()}: %{{y:,.0f}}<br>' +
                             '<extra></extra>'
            ))

            # Apply standard styling with smart Y-axis range
            self._apply_standard_style(fig,
                                       title or f"{metric.title()} Trend by {dimension.title()}",
                                       dimension.title(),
                                       metric.title(),
                                       show_legend=False,
                                       yaxis_range=yaxis_range)

        return fig
    
    def create_area_chart(self, data, metric, dimension, title=None):
        """Create a beautiful area chart from real data (supports multi-series with stacked areas)"""
        if data is None or data.empty:
            raise Exception("❌ No data available for area chart generation")

        # Reset index to ensure we're not plotting index values
        data = data.reset_index(drop=True)

        # Detect multi-series structure
        base_columns = self._get_base_columns(data)
        calculated_col = self._get_calculated_column(data)
        is_multi_series = len(data.columns) == 3 and any('group_by_' in str(col) for col in data.columns)
        
        # INTELLIGENT MULTI-SERIES DETECTION
        if not is_multi_series and len(base_columns) >= 2:
            dimension_col = base_columns[0]
            
            if self._detect_duplicate_dimensions(data, dimension_col):
                grouping_col = self._infer_grouping_column(data, dimension_col)
                
                if grouping_col:
                    print(f"⚠️  Detected multiple values per {dimension_col} - treating as multi-series grouped by {grouping_col}")
                    metric_col = calculated_col if calculated_col else base_columns[-1]
                    data = data[[dimension_col, grouping_col, metric_col]].copy()
                    is_multi_series = True
                else:
                    print(f"⚠️  Found duplicate time periods - aggregating by averaging...")
                    metric_col = calculated_col if calculated_col else base_columns[-1]
                    data = data.groupby(dimension_col)[metric_col].mean().reset_index()
                    is_multi_series = False

        if is_multi_series:
            # Multi-series area chart - THE PLOTLY EXPRESS WAY
            dimension_col = data.columns[0]
            group_by_col = data.columns[1]
            metric_col = data.columns[2]

            # Sort fiscal year data if detected (CRITICAL for area charts!)
            data = self._sort_fiscal_year_data(data, dimension_col)

            # Check for NaN values
            nan_count = data[metric_col].isna().sum()
            if nan_count > 0:
                print(f"⚠️  Found {nan_count} missing values in {metric_col} - will connect across gaps")

            num_areas = data[group_by_col].nunique()
            print(f"📊 Multi-series stacked area: {num_areas} areas over {len(data[dimension_col].unique())} periods")

            # Get extended color palette to avoid repetition
            colors = self._get_extended_color_palette(num_areas, 'professional')

            # Use Plotly Express - automatic area stacking by color parameter
            fig = px.area(
                data,
                x=dimension_col,
                y=metric_col,
                color=group_by_col,  # This does the loop for you!
                color_discrete_sequence=colors
            )

            # Style the areas with subtle opacity, connect across NaN gaps
            fig.update_traces(
                line=dict(width=0.5),
                connectgaps=True
            )

            # Apply standard styling
            legend_title = group_by_col.replace('group_by_', '').title()
            self._apply_standard_style(fig,
                                       title or f"{metric.title()} Trend by {dimension.title()} (stacked)",
                                       dimension.title(),
                                       metric.title(),
                                       show_legend=True,
                                       legend_title=legend_title)

            fig.update_layout(hovermode='x unified')

            print(f"   ✅ Created stacked area chart with {num_areas} areas using Plotly Express")

        else:
            # Single-series area chart
            dimension_col = data.columns[0]
            metric_col = data.columns[1]

            print(f"📊 Single-series area: {len(data)} data points")

            # Sort fiscal year data if detected (CRITICAL for area charts!)
            # This replaces the old quarter-only sorting logic
            data = self._sort_fiscal_year_data(data, dimension_col)

            # Check for NaN values
            nan_count = data[metric_col].isna().sum()
            if nan_count > 0:
                print(f"⚠️  Found {nan_count} missing values in {metric_col} - will connect across gaps")

            # Extract values explicitly
            x_values = data[dimension_col].tolist()
            y_values = data[metric_col].tolist()

            import plotly.graph_objects as go
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                name=metric.title(),
                line=dict(width=2, color='#3498DB'),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.3)',
                connectgaps=True,  # Connect across NaN values
                hovertemplate='<b>%{x}</b><br>' +
                             f'{metric.title()}: %{{y:,.0f}}<br>' +
                             '<extra></extra>'
            ))

            # Apply standard styling
            self._apply_standard_style(fig,
                                       title or f"{metric.title()} Trend by {dimension.title()}",
                                       dimension.title(),
                                       metric.title(),
                                       show_legend=False)

        return fig
    
    def create_pie_chart(self, data, metric, dimension, title=None):
        """Create a beautiful pie chart from real data with enhanced colors (supports multi-series with subplots)"""
        if data is None or data.empty:
            raise Exception("❌ No data available for pie chart generation")
        
        # Reset index to ensure we're not plotting index values
        data = data.reset_index(drop=True)
        
        # Check if this is multi-series data (3 columns: dimension, group_by, metric)
        is_multi_series = len(data.columns) == 3 and 'group_by_' in data.columns[1]
        
        if is_multi_series:
            # Multi-series pie chart: Create subplots for each group
            dimension_col = data.columns[0]
            group_by_col = data.columns[1]
            metric_col = data.columns[2]
            
            print(f"\n🔍 DEBUG create_pie_chart (MULTI-SERIES SUBPLOTS):")
            print(f"   Data shape: {data.shape}")
            print(f"   Dimension: {dimension_col}")
            print(f"   Group By: {group_by_col}")
            print(f"   Metric: {metric_col}")
            
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            categories = data[group_by_col].unique()
            num_categories = len(categories)
            
            # Limit to 4 subplots for readability
            if num_categories > 4:
                print(f"   ⚠️  Limiting to top 4 categories (out of {num_categories})")
                # Get top 4 categories by total metric value
                top_categories = data.groupby(group_by_col)[metric_col].sum().nlargest(4).index.tolist()
                data = data[data[group_by_col].isin(top_categories)]
                categories = top_categories
                num_categories = 4
            
            # Create subplots (2 columns max)
            cols = min(2, num_categories)
            rows = (num_categories + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'pie'}] * cols for _ in range(rows)],
                subplot_titles=[str(cat) for cat in categories]
            )
            
            colors = self.color_palettes['modern']
            
            for idx, category in enumerate(categories):
                row = idx // cols + 1
                col = idx % cols + 1
                
                category_data = data[data[group_by_col] == category]
                labels = category_data[dimension_col].tolist()
                values = category_data[metric_col].tolist()
                
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        name=str(category),
                        marker=dict(colors=colors, line=dict(color='white', width=2)),
                        hovertemplate='<b>%{label}</b><br>' +
                                     f'{metric.title()}: %{{value:,.0f}}<br>' +
                                     'Percentage: %{percent}<br>' +
                                     '<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title={
                    'text': title or f"{metric.title()} Distribution by {dimension.title()} (grouped by {group_by_col.replace('group_by_', '')})",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
                },
                font={'family': 'Arial', 'color': '#34495E', 'size': 12},
                height=300 * rows,
                showlegend=False,
                template='plotly_white'
            )
            
            print(f"   ✅ Created {num_categories} pie chart subplots")
            return fig
        
        # Single-series pie chart (original logic)
        # Handle case where pie chart has only one metric (single column data)
        if len(data.columns) == 1:
            print("📊 Pie chart with single metric - using row index as dimension")
            # For single column, use index as dimension
            metric_col = data.columns[0]
            data_for_chart = data.copy()
            data_for_chart['dimension'] = data_for_chart.index.astype(str)
            dimension_col = 'dimension'
        else:
            metric_col = data.columns[1]  # Second column is metric
            dimension_col = data.columns[0]  # First column is dimension
            data_for_chart = data

        # Data is already aggregated and filtered, but check cardinality for visualization
        num_slices = len(data_for_chart)
        if num_slices > 8:
            print(f"⚠️  Warning: Showing {num_slices} categories in pie chart. Consider using fewer categories for better readability.")

        # Extract values explicitly to avoid any Plotly confusion
        labels = data_for_chart[dimension_col].astype(str).tolist()
        values = data_for_chart[metric_col].tolist()

        print(f"   📊 Plotting pie chart with {num_slices} slices")

        # Get extended color palette to avoid repetition
        colors = self._get_extended_color_palette(num_slices, 'modern')

        # Create figure using graph_objects for more control
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         f'{metric.title()}: %{{value:,.0f}}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': title or f"{metric.title()} Distribution by {dimension.title()}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            font={'family': 'Arial', 'color': '#34495E', 'size': 12},
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            height=500,
            margin=dict(t=80, l=60, r=150, b=60),
            template='plotly_white'
        )

        # Enhance pie chart styling
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

        return fig
    
    def create_scatter_plot(self, data, metrics, dimension, title=None):
        """Create a scatter plot for comparing metrics (supports multi-series with colored groups)"""
        if data is None or data.empty:
            raise Exception("❌ No data available for scatter plot generation")
        
        # Reset index to ensure we're not plotting index values
        data = data.reset_index(drop=True)
        
        # Check if this is multi-series data (3 columns: dimension, group_by, metric)
        is_multi_series = len(data.columns) == 3 and 'group_by_' in data.columns[1]
        
        if is_multi_series:
            # For multi-series scatter - THE PLOTLY EXPRESS WAY
            dimension_col = data.columns[0]
            group_by_col = data.columns[1]
            metric_col = data.columns[2]

            num_groups = data[group_by_col].nunique()
            print(f"📊 Multi-series scatter: {num_groups} groups with {len(data)} points")

            # Get extended color palette to avoid repetition
            colors = self._get_extended_color_palette(num_groups, 'vibrant')

            fig = px.scatter(
                data,
                x=dimension_col,
                y=metric_col,
                color=group_by_col,  # This does the grouping for you!
                size=metric_col,
                color_discrete_sequence=colors
            )

            # Apply standard styling
            legend_title = group_by_col.replace('group_by_', '').title()
            self._apply_standard_style(fig,
                                       title or f"{metrics} by {dimension} (grouped by {legend_title})",
                                       dimension.title(),
                                       metrics if isinstance(metrics, str) else metrics[0] if metrics else 'Value',
                                       show_legend=True,
                                       legend_title=legend_title)

            print(f"   ✅ Created multi-series scatter plot with {num_groups} groups using Plotly Express")
            return fig

        # Single-series scatter plot (original logic)
        # For scatter plot, we need at least 2 numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            metric1_col = numeric_cols[0]
            metric2_col = numeric_cols[1]
            dimension_col = data.columns[0]  # First column as color dimension

            # Extract values explicitly to avoid any Plotly confusion
            x_values = data[metric1_col].tolist()
            y_values = data[metric2_col].tolist()

            print(f"   📊 Plotting scatter with {len(data)} points across {len(data[dimension_col].unique())} categories")

            # Create figure using graph_objects for more control
            import plotly.graph_objects as go

            fig = go.Figure()

            # Group by dimension for colored scatter points
            unique_dims = data[dimension_col].unique()
            num_dims = len(unique_dims)
            colors_palette = self._get_extended_color_palette(num_dims, 'vibrant')

            for idx, dim_value in enumerate(unique_dims):
                dim_data = data[data[dimension_col] == dim_value]
                x_dim = dim_data[metric1_col].tolist()
                y_dim = dim_data[metric2_col].tolist()

                fig.add_trace(go.Scatter(
                    x=x_dim,
                    y=y_dim,
                    mode='markers',
                    name=str(dim_value),
                    marker=dict(
                        size=10,
                        color=colors_palette[idx],
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{dim_value}</b><br>' +
                                 f'{metric1_col}: %{{x:,.0f}}<br>' +
                                 f'{metric2_col}: %{{y:,.0f}}<br>' +
                                 '<extra></extra>'
                ))

            # Apply standard styling with gridlines on both axes for scatter
            self._apply_standard_style(fig,
                                       title or f"{metric1_col} vs {metric2_col} by {dimension_col}",
                                       metric1_col,
                                       metric2_col,
                                       show_legend=True)

            # Override grid for scatter - show on both axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECF0F1')

            return fig
        else:
            # Fallback to bar chart if insufficient numeric columns
            return self.create_bar_chart(data, metrics[0] if metrics else 'value', dimension, title)
    
    def create_table(self, data, title=None):
        """Create a table visualization from real data"""
        if data is None or data.empty:
            raise Exception("❌ No data available for table generation")
        
        # Reset index to ensure clean data
        data = data.reset_index(drop=True)
        
        # Extract column names and values explicitly
        column_names = data.columns.tolist()
        column_values = [data[col].tolist() for col in data.columns]
        
        print(f"   📊 Creating table with columns: {column_names}")
        print(f"   📊 Table rows: {len(data)}")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=column_names,
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=column_values,
                fill_color='white',
                align='center',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=title or "Data Table",
            height=min(400 + len(data) * 20, 800)  # Dynamic height based on data
        )
        
        return fig
    
    def create_heatmap(self, data, metric, dimensions, title=None):
        """Create a heatmap from real data"""
        if data is None or data.empty:
            raise Exception("❌ No data available for heatmap generation")
        
        # Reset index to ensure clean data
        data = data.reset_index(drop=True)
            
        # For heatmap, we need to pivot the data if we have enough dimensions
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
            # Pivot data for heatmap
            try:
                pivot_data = data.pivot_table(
                    values=numeric_cols[0], 
                    index=categorical_cols[0], 
                    columns=categorical_cols[1], 
                    aggfunc='mean',
                    fill_value=0
                )
                
                # Extract values explicitly to avoid any Plotly confusion
                z_values = pivot_data.values.tolist()
                x_values = pivot_data.columns.tolist()
                y_values = pivot_data.index.tolist()
                
                print(f"   📊 Creating heatmap with shape: {len(y_values)} x {len(x_values)}")
                print(f"   📊 X categories: {x_values}")
                print(f"   📊 Y categories: {y_values}")
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=x_values,
                    y=y_values,
                    colorscale='Viridis',
                    hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>' +
                                 f'{metric.title()}: %{{z:,.2f}}<br>' +
                                 '<extra></extra>'
                ))
                
                fig.update_layout(
                    title={
                        'text': title or f"{metric.title()} Heatmap",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
                    },
                    xaxis_title=categorical_cols[1],
                    yaxis_title=categorical_cols[0],
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Arial', 'color': '#34495E'},
                    height=500,
                    margin=dict(t=80, l=100, r=60, b=60),
                    template='plotly_white'
                )
                
                return fig
            except Exception as e:
                print(f"⚠️  Heatmap creation failed: {e}, falling back to bar chart")
                # Fallback to bar chart if pivot fails
                return self.create_bar_chart(data, metric, dimensions[0], title)
        else:
            # Fallback to bar chart if insufficient categorical columns
            return self.create_bar_chart(data, metric, dimensions[0], title)
    
    def create_card(self, data, metric, aggregation=None, filters=None):
        """Create a single value card from real data"""
        if data is None or data.empty:
            raise Exception("❌ No data available for card generation")
        
        # Reset index to ensure clean data
        data = data.reset_index(drop=True)
            
        metric_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Calculate aggregated value and convert to Python native type
        if aggregation == 'sum' or aggregation == 'total':
            value = float(data[metric_col].sum())
        elif aggregation == 'avg' or aggregation == 'average':
            value = float(data[metric_col].mean())
        elif aggregation == 'count':
            value = float(len(data))
        elif aggregation == 'max':
            value = float(data[metric_col].max())
        elif aggregation == 'min':
            value = float(data[metric_col].min())
        else:
            value = float(data[metric_col].sum())  # Default to sum
        
        print(f"   📊 Card value ({aggregation or 'total'}): {value}")
            
        # Format the value
        if value >= 1000000:
            formatted_value = f"${value/1000000:.1f}M"
        elif value >= 1000:
            formatted_value = f"${value/1000:.1f}K"
        else:
            formatted_value = f"${value:.0f}"
            
        filter_text = f" ({', '.join(filters)})" if filters else ""
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            title={
                "text": f"{(aggregation or 'Total').title()} {metric.title()}{filter_text}",
                "font": {"size": 20}
            },
            number={"font": {"size": 40}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def detect_chart_type_from_query(self, query_text):
        """
        Detect chart type directly from query text keywords
        Priority 1: Direct keyword detection
        """
        if not query_text:
            return None
            
        query_lower = query_text.lower()
        
        # Chart type keywords mapping
        chart_keywords = {
            'bar': ['bar chart', 'bar graph', 'column chart', 'bar', 'bars', 'as bar'],
            'line': ['line chart', 'line graph', 'trend', 'line', 'lines', 'timeline', 'as line'],
            'pie': ['pie chart', 'pie graph', 'pie', 'donut', 'distribution', 'as pie'],
            'scatter': ['scatter plot', 'scatter chart', 'scatter', 'correlation', 'as scatter'],
            'area': ['area chart', 'area graph', 'as area'],
            'table': ['table', 'data table', 'tabular', 'list', 'as table'],
            'heatmap': ['heatmap', 'heat map', 'correlation matrix', 'as heatmap'],
            'card': ['card', 'kpi', 'metric card', 'indicator', 'single value', 'as card']
        }
        
        # Check for explicit chart type mentions
        for chart_type, keywords in chart_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    print(f"🎯 Chart type detected from query: {chart_type} (keyword: '{keyword}')")
                    return chart_type
        
        # Check for visualization-related words that might indicate chart preference
        if any(word in query_lower for word in ['visualize', 'plot', 'graph', 'chart']):
            # If generic visualization word but no specific type, return None to let other methods decide
            print("📊 Generic visualization word found, continuing to other detection methods")
        
        return None
    
    def generate_chart(self, entities, query_text=""):
        """
        Main method to generate chart based on extracted entities - NO HARDCODING
        entities: dict with keys like 'METRIC', 'DIMENSION', 'CHART_TYPE', 'FILTER', 'AGGREGATION'
                 OR lowercase keys 'metric', 'dimension', 'chart_type', etc. (SmartQueryParser format)
        """
        # Support both old NER format (uppercase keys with lists) and new SmartQueryParser format (lowercase keys with strings)
        # SmartQueryParser format: {'metric': 'sales', 'dimension': 'region', ...}
        # Old NER format: {'METRIC': ['sales'], 'DIMENSION': ['region'], ...}
        
        # Extract entity values - support both formats
        metrics = entities.get('METRIC', entities.get('metric', []))
        dimensions = entities.get('DIMENSION', entities.get('dimension', []))
        chart_types = entities.get('CHART_TYPE', entities.get('chart_type', []))
        filters = entities.get('FILTER', entities.get('filters', []))
        aggregations = entities.get('AGGREGATION', entities.get('aggregation', []))
        
        # Convert single string values to lists for uniform handling
        if isinstance(metrics, str):
            metrics = [metrics] if metrics else []
        if isinstance(dimensions, str):
            dimensions = [dimensions] if dimensions else []
        if isinstance(chart_types, str):
            chart_types = [chart_types] if chart_types else []
        if isinstance(aggregations, str):
            aggregations = [aggregations] if aggregations else []
        if isinstance(filters, str):
            filters = [filters] if filters else []
        
        # NEW: Extract calculation fields (from SmartQueryParser)
        calculation_type = entities.get('calculation_type')
        calculation_window = entities.get('calculation_window')
        comparison_type = entities.get('comparison_type')
        group_by = entities.get('group_by')  # NEW: Multi-series support
        time_granularity = entities.get('time_granularity')  # NEW: Time granularity
        
        if calculation_type:
            print(f"🧮 Calculation detected: {calculation_type}")
            if calculation_window:
                print(f"   Window size: {calculation_window}")
        
        if group_by:
            print(f"📊 Multi-series chart detected: group_by = {group_by}")
        if time_granularity:
            print(f"⏰ Time granularity: {time_granularity}")
        
        # Validate that we have essential entities OR column names in query
        # New approach: Try column-based even if NER finds nothing
        if not metrics and not dimensions and not query_text:
            raise Exception("❌ No metrics, dimensions, or query text provided. Cannot generate chart.")
        
        # Handle single values vs lists
        metric = metrics[0] if metrics else 'detected_metric'
        dimension = dimensions[0] if dimensions else 'detected_dimension'
        chart_type = chart_types[0] if chart_types else None
        aggregation = aggregations[0] if aggregations else None
        
        # Debug output
        print(f"🔍 Extracted from entities:")
        print(f"   Metric: {metric}")
        print(f"   Dimension: {dimension}")
        print(f"   Chart type: {chart_type}")
        print(f"   Aggregation: {aggregation}")
        
        # Special handling for multi-metric queries
        if len(metrics) > 1 and not dimension:
            # If we have multiple metrics but no dimension, use first metric as dimension
            dimension = metrics[1] if len(metrics) > 1 else 'category'
            print(f"🔄 Using second metric as dimension: {dimension}")
        elif not metric and not dimension and query_text:
            # Set defaults for column-based approach
            metric = 'auto_detected'
            dimension = 'auto_detected'
            print(f"🔄 Using column-based detection for: {query_text}")
        elif not dimension:
            dimension = 'category'
            print(f"🔄 Using default dimension: {dimension}")
        elif not metric:
            metric = 'value'
            print(f"🔄 Using default metric: {metric}")
        
        # Get real data - COLUMN-BASED APPROACH WITH GROUP_BY SUPPORT
        try:
            data = self.get_chart_data(
                metric, dimension, filters, aggregation, query_text, group_by,
                calculation_type, calculation_window
            )
            print(f"✅ Retrieved {len(data)} rows of real data")
            
            # Extract actual column names from data for title
            # Handle different data structures: single-series vs multi-series
            # Note: Data may have additional calculated columns (YoY_Growth_Pct, etc.) if calculation was applied
            
            # Check if data has calculated metric columns
            has_yoy = 'YoY_Growth_Pct' in data.columns
            has_mom = 'MoM_Change_Pct' in data.columns
            has_cumulative = 'Cumulative_Total' in data.columns
            has_ma = any('MA_' in col for col in data.columns)
            
            # Detect multi-series BEFORE calculation columns are added
            base_columns = [col for col in data.columns if not col in ['YoY_Growth_Pct', 'MoM_Change_Pct', 'Cumulative_Total', 'Previous_Value'] and not col.startswith('MA_')]
            is_multi_series = len(base_columns) == 3 and any('group_by_' in col for col in base_columns)
            
            if len(base_columns) == 1:
                # Single column data (for pie charts)
                actual_metric = base_columns[0].replace('metric_', '')
                actual_dimension = 'categories'
                print(f"📊 Using single metric: {actual_metric}")
            elif is_multi_series:
                # Multi-series data: dimension, group_by, metric
                actual_dimension = base_columns[0].replace('dimension_', '')
                actual_group_by = base_columns[1].replace('group_by_', '')
                actual_metric = base_columns[2].replace('metric_', '')
                print(f"📊 Multi-series detected: {actual_metric} (metric) × {actual_dimension} (dimension) × {actual_group_by} (group_by)")
            else:
                # Single-series data: dimension, metric
                actual_dimension = base_columns[0].replace('dimension_', '')
                actual_metric = base_columns[1].replace('metric_', '') if len(base_columns) >= 2 else 'value'
                print(f"📊 Using columns: {actual_metric} (metric) × {actual_dimension} (dimension)")
            
            # If calculation was applied in get_chart_data(), use the calculated column for visualization
            if calculation_type and calculation_type not in [None, 'none', 'None']:
                if has_yoy:
                    print(f"📈 Using calculated metric: YoY_Growth_Pct")
                    actual_metric = 'YoY_Growth_Pct'
                elif has_mom:
                    print(f"📈 Using calculated metric: MoM_Change_Pct")
                    actual_metric = 'MoM_Change_Pct'
                elif has_cumulative:
                    print(f"📈 Using calculated metric: Cumulative_Total")
                    actual_metric = 'Cumulative_Total'
                elif has_ma:
                    ma_col = [col for col in data.columns if col.startswith('MA_')][0]
                    print(f"📈 Using calculated metric: {ma_col}")
                    actual_metric = ma_col.replace('MA_', 'Moving Average ')

            
        except Exception as e:
            raise Exception(f"❌ Failed to get chart data: {e}")
        
        # PRIORITY SYSTEM FOR CHART TYPE DETECTION
        detected_chart_type = None
        
        print(f"\n🔍 CHART TYPE DETECTION DEBUG:")
        print(f"   chart_type variable: '{chart_type}' (type: {type(chart_type)})")
        print(f"   query_text: '{query_text}'")
        
        # Priority 1: Check query text for explicit chart type keywords
        query_chart_type = self.detect_chart_type_from_query(query_text)
        print(f"   Priority 1 result: '{query_chart_type}'")
        
        if query_chart_type:
            detected_chart_type = query_chart_type
            print(f"🎯 Priority 1 - Chart type from query: {detected_chart_type}")
        
        # Priority 2: Use SmartQueryParser extracted chart type if available and no query detection
        elif chart_type and chart_type not in ['detected_chart_type', 'auto', None]:
            detected_chart_type = chart_type
            print(f"🤖 Priority 2 - Chart type from SmartQueryParser: {detected_chart_type}")
        
        # Priority 3: Auto-detect based on data characteristics
        else:
            detected_chart_type = self.auto_detect_chart_type(metric, dimension, data)
            print(f"📊 Priority 3 - Auto-detected chart type: {detected_chart_type}")
        
        # Ensure we have a valid chart type
        if not detected_chart_type:
            detected_chart_type = 'bar'  # Final fallback
            print(f"🔧 Using final fallback chart type: {detected_chart_type}")
        
        # Use the detected chart type
        chart_type = detected_chart_type

        # Create title - prioritize LLM-provided title if available
        if '_title' in entities and entities['_title']:
            title = entities['_title']
        else:
            # Fallback to generated title
            filter_text = f" ({', '.join(filters)})" if filters else ""
            title = f"{query_text}" if query_text else f"{actual_metric} by {actual_dimension}{filter_text}"
        
        # Filter data to only include relevant columns for charting
        # This handles cases where calculated metrics (Cumulative_Total, YoY_Growth_Pct, etc.) were added
        chart_data = self._prepare_data_for_chart(data, actual_metric, is_multi_series, base_columns)
        
        # Generate appropriate chart based on type
        try:
            chart_type_lower = chart_type.lower() if chart_type else 'bar'
            
            if 'bar' in chart_type_lower:
                return self.create_bar_chart(chart_data, actual_metric, actual_dimension, title)
            elif 'line' in chart_type_lower:
                return self.create_line_chart(chart_data, actual_metric, actual_dimension, title)
            elif 'area' in chart_type_lower:
                # Use dedicated area chart method with proper multi-series support
                return self.create_area_chart(chart_data, actual_metric, actual_dimension, title)
            elif 'pie' in chart_type_lower:
                return self.create_pie_chart(chart_data, actual_metric, actual_dimension, title)
            elif 'scatter' in chart_type_lower:
                return self.create_scatter_plot(chart_data, [actual_metric], actual_dimension, title)
            elif 'table' in chart_type_lower:
                return self.create_table(chart_data, title)
            elif 'heatmap' in chart_type_lower:
                return self.create_heatmap(chart_data, actual_metric, [actual_dimension], title)
            elif 'card' in chart_type_lower:
                return self.create_card(chart_data, actual_metric, aggregation, filters)
            else:
                # Auto-detect appropriate chart type
                auto_chart_type = self.auto_detect_chart_type(actual_metric, actual_dimension, chart_data)
                print(f"🤖 Using auto-detected chart type: {auto_chart_type}")
                
                if auto_chart_type == 'bar':
                    return self.create_bar_chart(chart_data, actual_metric, actual_dimension, title)
                elif auto_chart_type == 'line':
                    return self.create_line_chart(chart_data, actual_metric, actual_dimension, title)
                elif auto_chart_type == 'pie':
                    return self.create_pie_chart(chart_data, actual_metric, actual_dimension, title)
                else:
                    return self.create_table(chart_data, title)
                    
        except Exception as e:
            raise Exception(f"❌ Failed to generate {chart_type} chart: {e}")

    def save_chart(self, fig, filename, format='html'):
        """Save chart to file"""
        try:
            if format.lower() == 'html':
                fig.write_html(f"{filename}.html")
            elif format.lower() == 'png':
                fig.write_image(f"{filename}.png")
            elif format.lower() == 'pdf':
                fig.write_image(f"{filename}.pdf")
            else:
                fig.write_html(f"{filename}.html")
            print(f"💾 Chart saved as {filename}.{format}")
        except Exception as e:
            print(f"❌ Error saving chart: {e}")
            
    def show_chart(self, fig):
        """Display chart"""
        try:
            fig.show()
        except Exception as e:
            print(f"❌ Error displaying chart: {e}")