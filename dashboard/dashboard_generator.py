import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys
import os
from datetime import datetime, timedelta
import json
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from charts.chart_generator import ChartGenerator

class ArchitectUIDashboard:
    """
    Modern ArchitectUI-inspired dashboard generator.
    Creates beautiful, responsive dashboards with multiple charts on a single page.
    Features clean design, smooth animations, and professional styling.
    """
    
    def __init__(self, theme="modern"):
        """
        Initialize the ArchitectUI dashboard generator
        
        Args:
            theme: Dashboard theme ('modern', 'dark', 'light', 'corporate')
        """
        self.chart_generator = ChartGenerator()
        self.charts = []
        self.chart_metadata = []
        self.theme = theme
        self.dashboard_id = str(uuid.uuid4())[:8]
        
        # ArchitectUI Color Schemes
        self.themes = {
            'modern': {
                'primary': '#3f6ad8',
                'secondary': '#764ba2',
                'background': '#f8fafc',
                'surface': '#ffffff',
                'text_primary': '#1e293b',
                'text_secondary': '#64748b',
                'accent': '#06b6d4',
                'success': '#10b981',
                'warning': '#f59e0b',
                'error': '#ef4444',
                'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            },
            'dark': {
                'primary': '#60a5fa',
                'secondary': '#a78bfa',
                'background': '#0f172a',
                'surface': '#1e293b',
                'text_primary': '#f1f5f9',
                'text_secondary': '#94a3b8',
                'accent': '#06b6d4',
                'success': '#10b981',
                'warning': '#f59e0b',
                'error': '#ef4444',
                'gradient': 'linear-gradient(135deg, #1e293b 0%, #334155 100%)'
            },
            'light': {
                'primary': '#2563eb',
                'secondary': '#7c3aed',
                'background': '#ffffff',
                'surface': '#f8fafc',
                'text_primary': '#0f172a',
                'text_secondary': '#475569',
                'accent': '#0891b2',
                'success': '#059669',
                'warning': '#d97706',
                'error': '#dc2626',
                'gradient': 'linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%)'
            }
        }
        
        self.current_theme = self.themes.get(theme, self.themes['modern'])
        print(f"🎨 ArchitectUI Dashboard initialized with {theme} theme")
    
    def add_chart_from_query(self, query, entities, chart_title=None):
        """
        Add a chart to the dashboard from a query and extracted entities
        
        Args:
            query: Natural language query string
            entities: Dictionary of extracted entities from NER
            chart_title: Optional custom title for the chart
        """
        try:
            print(f"📊 Adding chart: {query}")
            fig = self.chart_generator.generate_chart(entities, query)
            
            if fig:
                # Enhanced metadata for ArchitectUI
                metadata = {
                    'id': f"chart_{len(self.charts) + 1}",
                    'query': query,
                    'entities': entities,
                    'title': chart_title or self._generate_chart_title(query, entities),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': self._detect_chart_type(entities),
                    'metrics': entities.get('METRIC', []),
                    'dimensions': entities.get('DIMENSION', [])
                }
                
                self.charts.append(fig)
                self.chart_metadata.append(metadata)
                print(f"✅ Chart added successfully! Total: {len(self.charts)}")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error adding chart: {e}")
            return False
    
    def add_chart_figure(self, fig, title="Custom Chart", description=None):
        """
        Add a pre-generated chart figure to the dashboard
        
        Args:
            fig: Plotly figure object
            title: Chart title
            description: Optional chart description
        """
        metadata = {
            'id': f"chart_{len(self.charts) + 1}",
            'title': title,
            'description': description,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'custom',
            'source': 'manual'
        }
        
        self.charts.append(fig)
        self.chart_metadata.append(metadata)
        print(f"✅ Chart '{title}' added! Total: {len(self.charts)}")
    
    def create_sample_charts(self, num_charts=6):
        """
        Create sample charts with diverse visualizations using real Dataset.xlsx data
        
        Args:
            num_charts: Number of sample charts to create
        """
        print(f"🎨 Creating {num_charts} sample charts from Dataset.xlsx...")
        
        # Sample queries matching actual Dataset.xlsx columns:
        # ProductKey, OrderDateKey, CustomerKey, SalesTerritoryKey, OrderQuantity, 
        # UnitPrice, ExtendedAmount, ProductStandardCost, TaxAmt, Freight, etc.
        sample_queries = [
            {
                'query': 'Show ExtendedAmount by SalesTerritoryKey',
                'entities': {
                    'METRIC': ['ExtendedAmount'],
                    'DIMENSION': ['SalesTerritoryKey'],
                    'CHART_TYPE': ['bar'],
                    'FILTER': [],
                    'AGGREGATION': ['total']
                },
                'title': 'Total Sales by Territory'
            },
            {
                'query': 'Display OrderQuantity by ProductKey',
                'entities': {
                    'METRIC': ['OrderQuantity'],
                    'DIMENSION': ['ProductKey'],
                    'CHART_TYPE': ['bar'],
                    'FILTER': ['top 10'],
                    'AGGREGATION': ['total']
                },
                'title': 'Top 10 Products by Order Quantity'
            },
            {
                'query': 'Show UnitPrice distribution by ProductKey',
                'entities': {
                    'METRIC': ['UnitPrice'],
                    'DIMENSION': ['ProductKey'],
                    'CHART_TYPE': ['bar'],
                    'FILTER': ['top 8'],
                    'AGGREGATION': ['average']
                },
                'title': 'Average Unit Price by Product'
            },
            {
                'query': 'Display Freight by SalesTerritoryKey',
                'entities': {
                    'METRIC': ['Freight'],
                    'DIMENSION': ['SalesTerritoryKey'],
                    'CHART_TYPE': ['pie'],
                    'FILTER': [],
                    'AGGREGATION': ['total']
                },
                'title': 'Freight Distribution by Territory'
            },
            {
                'query': 'Show TaxAmt by CustomerKey',
                'entities': {
                    'METRIC': ['TaxAmt'],
                    'DIMENSION': ['CustomerKey'],
                    'CHART_TYPE': ['bar'],
                    'FILTER': ['top 10'],
                    'AGGREGATION': ['total']
                },
                'title': 'Top 10 Customers by Tax Amount'
            },
            {
                'query': 'Display ProductStandardCost by ProductKey',
                'entities': {
                    'METRIC': ['ProductStandardCost'],
                    'DIMENSION': ['ProductKey'],
                    'CHART_TYPE': ['bar'],
                    'FILTER': ['top 10'],
                    'AGGREGATION': ['average']
                },
                'title': 'Product Standard Cost Analysis'
            }
        ]
        
        # Add requested number of charts
        for i in range(min(num_charts, len(sample_queries))):
            query_data = sample_queries[i]
            try:
                self.add_chart_from_query(
                    query_data['query'], 
                    query_data['entities'],
                    query_data['title']
                )
            except Exception as e:
                print(f"⚠️  Could not create sample chart {i+1}: {e}")
                continue
    
    def generate_dashboard(self, title="ArchitectUI Analytics Dashboard", layout="auto", save_path=None):
        """
        Generate a beautiful ArchitectUI-inspired dashboard with responsive grid layout
        
        Args:
            title: Dashboard title
            layout: Layout style ('auto', 'grid', 'masonry', 'single-row', 'double-row')
            save_path: Path to save the dashboard HTML file
            
        Returns:
            HTML string containing the complete dashboard
        """
        if not self.charts:
            print("⚠️  No charts available. Add charts before generating dashboard.")
            return None
        
        num_charts = len(self.charts)
        print(f"🎨 Generating ArchitectUI dashboard with {num_charts} charts...")
        
        # Calculate optimal layout
        layout_config = self._calculate_layout(num_charts, layout)
        
        # Generate HTML dashboard
        dashboard_html = self._generate_html_dashboard(title, layout_config)
        
        # Save dashboard if path provided
        if save_path:
            self.save_dashboard_html(dashboard_html, save_path)
        
        print(f"✅ ArchitectUI dashboard generated successfully!")
        return dashboard_html
    
    def _calculate_layout(self, num_charts, layout_style):
        """Calculate optimal layout configuration for charts"""
        if layout_style == "auto":
            if num_charts <= 2:
                return {"type": "single-row", "cols": num_charts}
            elif num_charts <= 4:
                return {"type": "grid", "cols": 2, "rows": 2}
            elif num_charts <= 6:
                return {"type": "grid", "cols": 3, "rows": 2}
            else:
                return {"type": "masonry", "cols": 3}
        elif layout_style == "grid":
            cols = min(3, num_charts)
            rows = (num_charts + cols - 1) // cols
            return {"type": "grid", "cols": cols, "rows": rows}
        elif layout_style == "single-row":
            return {"type": "single-row", "cols": num_charts}
        elif layout_style == "double-row":
            cols = (num_charts + 1) // 2
            return {"type": "grid", "cols": cols, "rows": 2}
        else:
            return {"type": "masonry", "cols": 3}
    
    def _generate_html_dashboard(self, title, layout_config):
        """Generate complete HTML dashboard with ArchitectUI styling"""
        
        # Convert plotly figures to HTML
        chart_htmls = []
        for i, (fig, metadata) in enumerate(zip(self.charts, self.chart_metadata)):
            # Fix binary encoding issue: Convert numpy arrays to lists
            # This ensures charts render properly in all browsers
            import numpy as np
            new_traces = []
            for trace in fig.data:
                trace_dict = trace.to_plotly_json()
                # Convert any numpy arrays to Python lists
                for key in trace_dict:
                    if isinstance(trace_dict[key], np.ndarray):
                        trace_dict[key] = trace_dict[key].tolist()
                    elif hasattr(trace_dict[key], 'tolist'):
                        try:
                            trace_dict[key] = trace_dict[key].tolist()
                        except (AttributeError, TypeError) as e:
                            # Keep original value if conversion fails
                            pass
                new_traces.append(trace_dict)
            
            # Create new figure with list data (no binary encoding)
            clean_fig = go.Figure(data=new_traces, layout=fig.layout)
            
            chart_html = pio.to_html(
                clean_fig,
                include_plotlyjs=False,
                div_id=f"chart-{metadata['id']}",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                    'responsive': True
                },
                full_html=False
            )
            
            chart_htmls.append({
                'html': chart_html,
                'metadata': metadata,
                'index': i
            })
        
        # Generate dashboard statistics
        stats = self._generate_dashboard_stats()
        
        # Create complete HTML
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    {self._get_architectui_styles()}
</head>
<body>
    <div class="architect-dashboard">
        <!-- Header Section -->
        <header class="dashboard-header">
            <div class="header-content">
                <div class="header-main">
                    <h1 class="dashboard-title">
                        <span class="title-icon">📊</span>
                        {title}
                    </h1>
                    <p class="dashboard-subtitle">Real-time analytics and insights</p>
                </div>
                <div class="header-actions">
                    <button class="action-btn" onclick="refreshDashboard()">
                        <span>🔄</span> Refresh
                    </button>
                    <button class="action-btn" onclick="exportDashboard()">
                        <span>📥</span> Export
                    </button>
                </div>
            </div>
        </header>

        <!-- Stats Overview -->
        <section class="stats-section">
            <div class="stats-grid">
                {self._generate_stats_cards(stats)}
            </div>
        </section>

        <!-- Charts Section -->
        <main class="charts-section">
            <div class="charts-container {layout_config['type']}-layout">
                {self._generate_chart_cards(chart_htmls)}
            </div>
        </main>

        <!-- Footer -->
        <footer class="dashboard-footer">
            <div class="footer-content">
                <p>🚀 Powered by ArchitectUI Dashboard • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Dashboard ID: {self.dashboard_id} • Theme: {self.theme.title()}</p>
            </div>
        </footer>
    </div>

    {self._get_dashboard_scripts()}
</body>
</html>"""
        
        return html_template
    
    def _generate_dashboard_stats(self):
        """Generate dashboard statistics"""
        if not self.chart_metadata:
            return {}
        
        chart_types = {}
        metrics = set()
        dimensions = set()
        
        for metadata in self.chart_metadata:
            chart_type = metadata.get('type', 'unknown')
            chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
            
            if 'metrics' in metadata:
                metrics.update(metadata['metrics'])
            if 'dimensions' in metadata:
                dimensions.update(metadata['dimensions'])
        
        return {
            'total_charts': len(self.charts),
            'chart_types': chart_types,
            'unique_metrics': len(metrics),
            'unique_dimensions': len(dimensions),
            'last_updated': datetime.now().strftime("%H:%M:%S")
        }
    
    def _generate_stats_cards(self, stats):
        """Generate HTML for statistics cards"""
        if not stats:
            return ""
        
        cards_html = f"""
            <div class="stat-card primary">
                <div class="stat-value">{stats['total_charts']}</div>
                <div class="stat-label">Total Charts</div>
                <div class="stat-icon">📊</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{stats['unique_metrics']}</div>
                <div class="stat-label">Metrics Analyzed</div>
                <div class="stat-icon">📈</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{stats['unique_dimensions']}</div>
                <div class="stat-label">Dimensions</div>
                <div class="stat-icon">🎯</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{stats['last_updated']}</div>
                <div class="stat-label">Last Updated</div>
                <div class="stat-icon">🕒</div>
            </div>
        """
        return cards_html
    
    def _generate_chart_cards(self, chart_htmls):
        """Generate HTML for chart cards"""
        cards_html = ""
        
        for chart_data in chart_htmls:
            metadata = chart_data['metadata']
            chart_html = chart_data['html']
            
            card_html = f"""
            <div class="chart-card" data-chart-id="{metadata['id']}">
                <div class="chart-header">
                    <h3 class="chart-title">{metadata['title']}</h3>
                    <div class="chart-actions">
                        <button class="chart-action" onclick="toggleFullscreen('{metadata['id']}')" title="Fullscreen">
                            ⛶
                        </button>
                        <button class="chart-action" onclick="downloadChart('{metadata['id']}')" title="Download">
                            ⬇
                        </button>
                    </div>
                </div>
                <div class="chart-content">
                    {chart_html}
                </div>
                <div class="chart-footer">
                    <span class="chart-type">{metadata.get('type', 'chart').title()}</span>
                    <span class="chart-timestamp">{metadata['timestamp']}</span>
                </div>
            </div>
            """
            cards_html += card_html
        
        return cards_html
    
    def _get_architectui_styles(self):
        """Get complete ArchitectUI-inspired CSS styles"""
        theme = self.current_theme
        
        return f"""
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            :root {{
                --primary: {theme['primary']};
                --secondary: {theme['secondary']};
                --background: {theme['background']};
                --surface: {theme['surface']};
                --text-primary: {theme['text_primary']};
                --text-secondary: {theme['text_secondary']};
                --accent: {theme['accent']};
                --success: {theme['success']};
                --warning: {theme['warning']};
                --error: {theme['error']};
                --gradient: {theme['gradient']};
                
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                
                --border-radius: 12px;
                --border-radius-lg: 16px;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: var(--background);
                color: var(--text-primary);
                line-height: 1.6;
                overflow-x: hidden;
            }}
            
            .architect-dashboard {{
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }}
            
            /* Header Styles */
            .dashboard-header {{
                background: var(--gradient);
                padding: 2rem 0;
                box-shadow: var(--shadow-lg);
                position: relative;
                overflow: hidden;
            }}
            
            .dashboard-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="20" fill="url(%23grid)"/></svg>');
                opacity: 0.3;
            }}
            
            .header-content {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 2rem;
                position: relative;
                z-index: 2;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .dashboard-title {{
                font-size: 2.5rem;
                font-weight: 700;
                color: white;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .title-icon {{
                font-size: 2rem;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
            }}
            
            .dashboard-subtitle {{
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                font-weight: 400;
            }}
            
            .header-actions {{
                display: flex;
                gap: 1rem;
            }}
            
            .action-btn {{
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: var(--border-radius);
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .action-btn:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }}
            
            /* Stats Section */
            .stats-section {{
                padding: 2rem 0;
                background: var(--surface);
            }}
            
            .stats-grid {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 2rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
            }}
            
            .stat-card {{
                background: white;
                padding: 2rem;
                border-radius: var(--border-radius-lg);
                box-shadow: var(--shadow-md);
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                border-left: 4px solid var(--primary);
            }}
            
            .stat-card:hover {{
                transform: translateY(-4px);
                box-shadow: var(--shadow-xl);
            }}
            
            .stat-card.primary {{ border-left-color: var(--primary); }}
            .stat-card.success {{ border-left-color: var(--success); }}
            .stat-card.warning {{ border-left-color: var(--warning); }}
            .stat-card.info {{ border-left-color: var(--accent); }}
            
            .stat-value {{
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
            }}
            
            .stat-label {{
                color: var(--text-secondary);
                font-weight: 500;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stat-icon {{
                position: absolute;
                top: 1.5rem;
                right: 1.5rem;
                font-size: 2rem;
                opacity: 0.3;
            }}
            
            /* Charts Section */
            .charts-section {{
                flex: 1;
                padding: 2rem 0;
                background: var(--background);
            }}
            
            .charts-container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 2rem;
            }}
            
            .grid-layout {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 2rem;
            }}
            
            .masonry-layout {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 2rem;
                grid-template-rows: masonry;
            }}
            
            .single-row-layout {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
            }}
            
            .chart-card {{
                background: white;
                border-radius: var(--border-radius-lg);
                box-shadow: var(--shadow-md);
                overflow: hidden;
                transition: all 0.3s ease;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }}
            
            .chart-card:hover {{
                transform: translateY(-4px);
                box-shadow: var(--shadow-xl);
            }}
            
            .chart-header {{
                padding: 1.5rem 2rem 1rem;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(135deg, rgba(63, 106, 216, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            }}
            
            .chart-title {{
                font-size: 1.25rem;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
            }}
            
            .chart-actions {{
                display: flex;
                gap: 0.5rem;
            }}
            
            .chart-action {{
                background: none;
                border: none;
                color: var(--text-secondary);
                font-size: 1.2rem;
                cursor: pointer;
                padding: 0.5rem;
                border-radius: 6px;
                transition: all 0.2s ease;
            }}
            
            .chart-action:hover {{
                background: rgba(63, 106, 216, 0.1);
                color: var(--primary);
            }}
            
            .chart-content {{
                padding: 1rem;
                min-height: 400px;
            }}
            
            .chart-footer {{
                padding: 1rem 2rem;
                background: rgba(0, 0, 0, 0.02);
                border-top: 1px solid rgba(0, 0, 0, 0.05);
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.85rem;
                color: var(--text-secondary);
            }}
            
            .chart-type {{
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 0.25rem 0.75rem;
                background: var(--primary);
                color: white;
                border-radius: 20px;
                font-size: 0.75rem;
            }}
            
            /* Footer */
            .dashboard-footer {{
                background: var(--text-primary);
                color: rgba(255, 255, 255, 0.8);
                padding: 2rem 0;
                text-align: center;
            }}
            
            .footer-content p {{
                margin: 0.5rem 0;
                font-size: 0.9rem;
            }}
            
            /* Responsive Design */
            @media (max-width: 768px) {{
                .header-content {{
                    flex-direction: column;
                    gap: 1.5rem;
                    text-align: center;
                }}
                
                .dashboard-title {{
                    font-size: 2rem;
                }}
                
                .stats-grid {{
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                }}
                
                .chart-card {{
                    margin: 0 -1rem;
                    border-radius: 0;
                }}
                
                .grid-layout,
                .masonry-layout,
                .single-row-layout {{
                    grid-template-columns: 1fr;
                    gap: 1rem;
                }}
            }}
            
            /* Animations */
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            .chart-card {{
                animation: fadeInUp 0.6s ease forwards;
            }}
            
            .stat-card {{
                animation: fadeInUp 0.4s ease forwards;
            }}
            
            /* Plotly overrides */
            .plotly-graph-div {{
                width: 100% !important;
                height: 100% !important;
            }}
            
            .modebar {{
                opacity: 0;
                transition: opacity 0.3s ease;
            }}
            
            .chart-content:hover .modebar {{
                opacity: 1;
            }}
        </style>
        """
    
    def _get_dashboard_scripts(self):
        """Get JavaScript functions for dashboard interactivity"""
        return """
        <script>
            // Dashboard functionality
            function refreshDashboard() {
                location.reload();
            }
            
            function exportDashboard() {
                window.print();
            }
            
            function toggleFullscreen(chartId) {
                const chartCard = document.querySelector(`[data-chart-id="${chartId}"]`);
                if (chartCard.requestFullscreen) {
                    chartCard.requestFullscreen();
                } else if (chartCard.webkitRequestFullscreen) {
                    chartCard.webkitRequestFullscreen();
                } else if (chartCard.mozRequestFullScreen) {
                    chartCard.mozRequestFullScreen();
                }
            }
            
            function downloadChart(chartId) {
                const plotlyDiv = document.querySelector(`#chart-${chartId}`);
                if (plotlyDiv) {
                    Plotly.downloadImage(plotlyDiv, {
                        format: 'png',
                        width: 1200,
                        height: 800,
                        filename: `chart-${chartId}`
                    });
                }
            }
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                console.log('ArchitectUI Dashboard loaded successfully');
                
                // Add smooth scrolling
                document.documentElement.style.scrollBehavior = 'smooth';
                
                // Stagger animation for chart cards
                const chartCards = document.querySelectorAll('.chart-card');
                chartCards.forEach((card, index) => {
                    card.style.animationDelay = `${index * 0.1}s`;
                });
                
                // Responsive plotly charts
                window.addEventListener('resize', function() {
                    const plotlyDivs = document.querySelectorAll('.plotly-graph-div');
                    plotlyDivs.forEach(div => {
                        Plotly.Plots.resize(div);
                    });
                });
            });
        </script>
        """
    
    def save_dashboard_html(self, html_content, filename="architectui_dashboard.html"):
        """
        Save dashboard HTML to file
        
        Args:
            html_content: Complete HTML content string
            filename: Output filename
        """
        try:
            if not filename.endswith('.html'):
                filename += '.html'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"💾 ArchitectUI dashboard saved to: {filename}")
            
            # Get absolute path
            abs_path = os.path.abspath(filename)
            print(f"📂 Full path: {abs_path}")
            
        except Exception as e:
            print(f"❌ Error saving dashboard: {e}")
    
    def show_dashboard(self, html_content):
        """
        Display dashboard in browser by saving temp file and opening
        
        Args:
            html_content: HTML content to display
        """
        try:
            import tempfile
            import webbrowser
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                f.write(html_content)
                temp_path = f.name
            
            # Open in browser
            webbrowser.open(f'file://{temp_path}')
            print(f"🌐 Dashboard opened in browser")
            
        except Exception as e:
            print(f"❌ Error displaying dashboard: {e}")
    
    def clear_charts(self):
        """Clear all charts from the dashboard"""
        self.charts = []
        self.chart_metadata = []
        print("🗑️  All charts cleared from dashboard")
    
    def get_chart_count(self):
        """Get the number of charts in the dashboard"""
        return len(self.charts)
    
    def print_summary(self):
        """Print a comprehensive summary of the dashboard contents"""
        print("\n" + "="*70)
        print(f"📊 ARCHITECTUI DASHBOARD SUMMARY")
        print("="*70)
        print(f"Dashboard ID: {self.dashboard_id}")
        print(f"Theme: {self.theme.title()}")
        print(f"Total Charts: {len(self.charts)}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.chart_metadata:
            print(f"\n📈 Chart Details:")
            print("-" * 70)
            
            for i, metadata in enumerate(self.chart_metadata, 1):
                print(f"\n{i:2d}. {metadata['title']}")
                print(f"     Type: {metadata.get('type', 'unknown').title()}")
                print(f"     Created: {metadata['timestamp']}")
                if 'metrics' in metadata and metadata['metrics']:
                    print(f"     Metrics: {', '.join(metadata['metrics'])}")
                if 'dimensions' in metadata and metadata['dimensions']:
                    print(f"     Dimensions: {', '.join(metadata['dimensions'])}")
        
        print("="*70 + "\n")
    
    def _generate_chart_title(self, query, entities):
        """Generate a descriptive chart title from query and entities"""
        metrics = entities.get('METRIC', [])
        dimensions = entities.get('DIMENSION', [])
        chart_type = entities.get('CHART_TYPE', ['visualization'])
        
        if metrics and dimensions:
            return f"{' & '.join(metrics).title()} by {' & '.join(dimensions).title()}"
        elif metrics:
            return f"{' & '.join(metrics).title()} Analysis"
        elif dimensions:
            return f"{' & '.join(dimensions).title()} Breakdown"
        else:
            return query.title()
    
    def _detect_chart_type(self, entities):
        """Detect chart type from entities"""
        chart_types = entities.get('CHART_TYPE', [])
        if chart_types:
            return chart_types[0].lower()
        return 'unknown'


# For backward compatibility, create an alias
DashboardGenerator = ArchitectUIDashboard


def main():
    """Main function for testing ArchitectUI dashboard generator"""
    print("🚀 ArchitectUI Dashboard Generator")
    print("="*60)
    print("📂 Using REAL DATA from /data/Dataset.xlsx")
    print("🎨 Modern, responsive dashboard with professional styling")
    print("📊 Multiple chart types with dynamic layouts")
    print("✨ Beautiful animations and interactions")
    print("="*60)
    
    # Create ArchitectUI dashboard generator
    dashboard = ArchitectUIDashboard(theme="modern")
    
    # Create sample charts using real data from Dataset.xlsx
    print("\n📊 Creating charts from real Dataset.xlsx data...")
    dashboard.create_sample_charts(num_charts=6)
    
    # Print summary
    dashboard.print_summary()
    
    # Generate and save dashboard
    print("\n🎨 Generating ArchitectUI dashboard...")
    html_content = dashboard.generate_dashboard(
        title="ArchitectUI Analytics Dashboard - Real Data",
        layout="auto",
        save_path="architectui_dashboard.html"
    )
    
    if html_content:
        print("\n✅ ArchitectUI dashboard created successfully!")
        print("\n💡 Key Features:")
        print("   📂 Real data from Dataset.xlsx (54,771 rows)")
        print("   ✨ Modern ArchitectUI-inspired design")
        print("   📊 Responsive grid layout with multiple charts")
        print("   🎨 Professional color scheme and animations")
        print("   📱 Mobile-friendly responsive design")
        print("   🔄 Interactive controls and fullscreen mode")
        print("   📥 Export and download capabilities")
        print("\n📂 Open architectui_dashboard.html in your browser to view")
        
        # Show in browser
        try:
            dashboard.show_dashboard(html_content)
        except Exception as e:
            print(f"\n⚠️  Could not auto-open browser: {e}")
            print("Please open architectui_dashboard.html manually.")
    else:
        print("❌ Failed to create dashboard")


if __name__ == "__main__":
    main()