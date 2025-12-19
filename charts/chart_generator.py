import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import data_connector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_connector import DataConnector

class ChartGenerator:
    """Dynamic chart generator based on NER extracted entities - NO HARDCODING"""
    
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
    
    def get_chart_data(self, metric, dimension, filters=None, aggregation=None, query_text=""):
        """
        Get data for chart generation - COLUMN-BASED APPROACH
        Priority: Actual column names in query > NER entities
        """
        if not self.data_connector:
            raise Exception("❌ No data connector available. Cannot generate charts without real data.")
            
        try:
            # Use the new column-based approach
            data = self.data_connector.get_data_for_chart_column_based(metric, dimension, filters, aggregation, query_text)
            if data is None or data.empty:
                raise Exception(f"❌ No data found for query: {query_text}")
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
        """Create a beautiful bar chart from real data with enhanced colors"""
        if data is None or data.empty:
            raise Exception("❌ No data available for bar chart generation")
            
        metric_col = data.columns[1]  # Second column is metric
        dimension_col = data.columns[0]  # First column is dimension
        
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
        
        # Choose color palette based on number of categories
        if unique_dims <= 7:
            colors = self.color_palettes['vibrant']
        elif unique_dims <= 15:
            colors = self.color_palettes['professional']
        else:
            colors = self.color_palettes['business']
        
        # Create color column for proper color assignment
        color_cycle = colors * (unique_dims // len(colors) + 1)
        aggregated_data['color'] = color_cycle[:unique_dims]
        
        fig = px.bar(
            aggregated_data, 
            x=dimension_col, 
            y=metric_col,
            color='color',
            title=title or f"{metric.title()} by {dimension.title()}",
            color_discrete_map={color: color for color in colors},
            template='plotly_white'
        )
        
        fig.update_layout(
            title={
                'text': title or f"{metric.title()} by {dimension.title()}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            xaxis_title=dimension.title(),
            yaxis_title=metric.title(),
            showlegend=False,  # Hide legend since we're using colors for aesthetics
            xaxis_tickangle=-45,
            margin=dict(b=120, t=80, l=60, r=60),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial', 'color': '#34495E'},
            height=500
        )
        
        # Add hover effects and styling
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                         f'{metric.title()}: %{{y:,.0f}}<br>' +
                         '<extra></extra>',
            marker_line_width=1,
            marker_line_color='white'
        )
        
        # Enhance gridlines
        fig.update_xaxes(
            showgrid=False,
            linecolor='#BDC3C7',
            linewidth=2
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            linecolor='#BDC3C7',
            linewidth=2
        )
        
        return fig
    
    def create_line_chart(self, data, metric, dimension, title=None):
        """Create a beautiful line chart from real data with enhanced colors"""
        if data is None or data.empty:
            raise Exception("❌ No data available for line chart generation")
            
        metric_col = data.columns[1]  # Second column is metric
        dimension_col = data.columns[0]  # First column is dimension
        
        fig = px.line(
            data, 
            x=dimension_col, 
            y=metric_col,
            title=title or f"{metric.title()} Trend by {dimension.title()}",
            markers=True,
            template='plotly_white',
            color_discrete_sequence=['#3498DB']
        )
        
        fig.update_layout(
            title={
                'text': title or f"{metric.title()} Trend by {dimension.title()}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            xaxis_title=dimension.title(),
            yaxis_title=metric.title(),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial', 'color': '#34495E'},
            height=500,
            margin=dict(t=80, l=60, r=60, b=60)
        )
        
        # Enhance line styling
        fig.update_traces(
            line=dict(width=3, color='#3498DB'),
            marker=dict(size=8, color='#E74C3C', line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>' +
                         f'{metric.title()}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        )
        
        # Enhance gridlines
        fig.update_xaxes(
            showgrid=False,
            linecolor='#BDC3C7',
            linewidth=2
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            linecolor='#BDC3C7',
            linewidth=2
        )
        
        return fig
    
    def create_pie_chart(self, data, metric, dimension, title=None):
        """Create a beautiful pie chart from real data with enhanced colors"""
        if data is None or data.empty:
            raise Exception("❌ No data available for pie chart generation")
        
        # Handle case where pie chart has only one metric (single column data)
        if len(data.columns) == 1:
            print("📊 Pie chart with single metric - using row index as dimension")
            # For single column, use index as dimension
            metric_col = data.columns[0]
            data_for_chart = data.copy()
            data_for_chart['dimension'] = data_for_chart.index
            dimension_col = 'dimension'
        else:
            metric_col = data.columns[1]  # Second column is metric
            dimension_col = data.columns[0]  # First column is dimension
            data_for_chart = data
        
        # Data is already aggregated and filtered, but check cardinality for visualization
        if len(data_for_chart) > 8:
            print(f"⚠️  Warning: Showing {len(data_for_chart)} categories in pie chart. Consider using fewer categories for better readability.")
        
        fig = px.pie(
            data_for_chart, 
            values=metric_col, 
            names=dimension_col,
            title=title or f"{metric.title()} Distribution by {dimension.title()}",
            color_discrete_sequence=self.color_palettes['modern'],
            template='plotly_white'
        )
        
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
            margin=dict(t=80, l=60, r=150, b=60)
        )
        
        # Enhance pie chart styling
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         f'{metric.title()}: %{{value:,.0f}}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(line=dict(color='white', width=2))
        )
        
        return fig
    
    def create_scatter_plot(self, data, metrics, dimension, title=None):
        """Create a scatter plot for comparing metrics"""
        if data is None or data.empty:
            raise Exception("❌ No data available for scatter plot generation")
            
        # For scatter plot, we need at least 2 numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            metric1_col = numeric_cols[0]
            metric2_col = numeric_cols[1]
            dimension_col = data.columns[0]  # First column as color dimension
            
            fig = px.scatter(
                data, 
                x=metric1_col, 
                y=metric2_col,
                color=dimension_col,
                title=title or f"{metric1_col} vs {metric2_col} by {dimension_col}",
                size_max=15
            )
            
            fig.update_layout(
                xaxis_title=metric1_col,
                yaxis_title=metric2_col
            )
            
            return fig
        else:
            # Fallback to bar chart if insufficient numeric columns
            return self.create_bar_chart(data, metrics[0] if metrics else 'value', dimension, title)
    
    def create_table(self, data, title=None):
        """Create a table visualization from real data"""
        if data is None or data.empty:
            raise Exception("❌ No data available for table generation")
            
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(data.columns),
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[data[col] for col in data.columns],
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
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis'
                ))
                
                fig.update_layout(
                    title=title or f"{metric.title()} Heatmap",
                    xaxis_title=categorical_cols[1],
                    yaxis_title=categorical_cols[0]
                )
                
                return fig
            except:
                # Fallback to bar chart if pivot fails
                return self.create_bar_chart(data, metric, dimensions[0], title)
        else:
            # Fallback to bar chart if insufficient categorical columns
            return self.create_bar_chart(data, metric, dimensions[0], title)
    
    def create_card(self, data, metric, aggregation=None, filters=None):
        """Create a single value card from real data"""
        if data is None or data.empty:
            raise Exception("❌ No data available for card generation")
            
        metric_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        if aggregation == 'sum' or aggregation == 'total':
            value = data[metric_col].sum()
        elif aggregation == 'avg' or aggregation == 'average':
            value = data[metric_col].mean()
        elif aggregation == 'count':
            value = len(data)
        elif aggregation == 'max':
            value = data[metric_col].max()
        elif aggregation == 'min':
            value = data[metric_col].min()
        else:
            value = data[metric_col].sum()  # Default to sum
            
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
        """
        # Extract entity values
        metrics = entities.get('METRIC', [])
        dimensions = entities.get('DIMENSION', [])
        chart_types = entities.get('CHART_TYPE', [])
        filters = entities.get('FILTER', [])
        aggregations = entities.get('AGGREGATION', [])
        
        # Validate that we have essential entities OR column names in query
        # New approach: Try column-based even if NER finds nothing
        if not metrics and not dimensions and not query_text:
            raise Exception("❌ No metrics, dimensions, or query text provided. Cannot generate chart.")
        
        # Handle single values vs lists
        metric = metrics[0] if metrics else 'detected_metric'
        dimension = dimensions[0] if dimensions else 'detected_dimension'
        chart_type = chart_types[0] if chart_types else None
        aggregation = aggregations[0] if aggregations else None
        
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
        
        # Get real data - COLUMN-BASED APPROACH
        try:
            data = self.get_chart_data(metric, dimension, filters, aggregation, query_text)
            print(f"✅ Retrieved {len(data)} rows of real data")
            
            # Extract actual column names from data for title
            # Handle single column data (for pie charts with one metric)
            if len(data.columns) == 1:
                actual_metric = data.columns[0].replace('metric_', '')
                actual_dimension = 'categories'
                print(f"📊 Using single metric: {actual_metric}")
            else:
                actual_dimension = data.columns[0].replace('dimension_', '')
                actual_metric = data.columns[1].replace('metric_', '')
                print(f"📊 Using columns: {actual_metric} (metric) × {actual_dimension} (dimension)")
            
        except Exception as e:
            raise Exception(f"❌ Failed to get chart data: {e}")
        
        # PRIORITY SYSTEM FOR CHART TYPE DETECTION
        detected_chart_type = None
        
        # Priority 1: Check query text for explicit chart type keywords
        query_chart_type = self.detect_chart_type_from_query(query_text)
        if query_chart_type:
            detected_chart_type = query_chart_type
            print(f"🎯 Priority 1 - Chart type from query: {detected_chart_type}")
        
        # Priority 2: Use NER extracted chart type if available and no query detection
        elif chart_type:
            detected_chart_type = chart_type
            print(f"🤖 Priority 2 - Chart type from NER: {detected_chart_type}")
        
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
        
        # Create title using actual column names
        filter_text = f" ({', '.join(filters)})" if filters else ""
        title = f"{query_text}" if query_text else f"{actual_metric} by {actual_dimension}{filter_text}"
        
        # Generate appropriate chart based on type
        try:
            chart_type_lower = chart_type.lower() if chart_type else 'bar'
            
            if 'bar' in chart_type_lower:
                return self.create_bar_chart(data, actual_metric, actual_dimension, title)
            elif 'line' in chart_type_lower:
                return self.create_line_chart(data, actual_metric, actual_dimension, title)
            elif 'area' in chart_type_lower:
                # For area chart, use line chart with filled area
                fig = self.create_line_chart(data, actual_metric, actual_dimension, title)
                fig.update_traces(fill='tozeroy')
                return fig
            elif 'pie' in chart_type_lower:
                return self.create_pie_chart(data, actual_metric, actual_dimension, title)
            elif 'scatter' in chart_type_lower:
                return self.create_scatter_plot(data, [actual_metric], actual_dimension, title)
            elif 'table' in chart_type_lower:
                return self.create_table(data, title)
            elif 'heatmap' in chart_type_lower:
                return self.create_heatmap(data, actual_metric, [actual_dimension], title)
            elif 'card' in chart_type_lower:
                return self.create_card(data, actual_metric, aggregation, filters)
            else:
                # Auto-detect appropriate chart type
                auto_chart_type = self.auto_detect_chart_type(actual_metric, actual_dimension, data)
                print(f"🤖 Using auto-detected chart type: {auto_chart_type}")
                
                if auto_chart_type == 'bar':
                    return self.create_bar_chart(data, actual_metric, actual_dimension, title)
                elif auto_chart_type == 'line':
                    return self.create_line_chart(data, actual_metric, actual_dimension, title)
                elif auto_chart_type == 'pie':
                    return self.create_pie_chart(data, actual_metric, actual_dimension, title)
                else:
                    return self.create_table(data, title)
                    
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