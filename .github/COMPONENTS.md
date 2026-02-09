# Gen-Dash Core Components

## 1. NLUChartPipeline (Query Processing)

**Location**: [nlu/chart_pipeline.py](nlu/chart_pipeline.py)

**Architecture**: Complete pipeline from natural language → entities → chart:

```python
Query → SmartQueryParser → ChartGenerator → Plotly Figure
```

**Usage Pattern**:

```python
pipeline = NLUChartPipeline(use_llm=True)
entities = pipeline.process_query("sales by region")
fig = pipeline.chart_generator.generate_chart(entities, query)
```

## 2. SmartQueryParser (Hybrid NLP)

**Location**: [nlu/smart_query_parser.py](nlu/smart_query_parser.py)

**Pattern**: Fuzzy matching first (70-80% queries, <50ms), LLM fallback (20-30%, ~500ms)

### Entity Structure

```python
# parse_query returns flat dict with lowercase keys
entities = {
    'metric': 'sales',                      # Numeric column to measure
    'dimension': 'region',                  # Categorical column to group by
    'chart_type': 'bar',                    # bar, line, pie, area, scatter, heatmap, radar
    'aggregation': 'sum',                   # sum, avg, count, min, max
    'filters': [],                          # Query filter conditions
    'time_period': '2023',                  # Optional time filter
    'limit': 10,                            # Optional result limit (e.g., top 10)
    # v1.2.0: Calculated metrics & multi-series support
    'calculation_type': 'yoy_growth',       # yoy_growth, mom_change, cumulative, moving_average, per_unit, percent_change
    'calculation_window': 3,                # For moving averages (e.g., 3-month MA)
    'group_by': 'category',                 # Secondary dimension for multi-series line/area charts
    'time_granularity': 'monthly',          # daily, monthly, quarterly, yearly
    'comparison_type': 'vs_previous'        # vs_previous, vs_baseline, year_over_year
}
```

### Hybrid Approach

1. Fuzzy matching tries to match query terms to dataset columns (fast)
2. If confidence < 70 or ambiguous, falls back to LLM (Claude 3 Haiku via OpenRouter)
3. Results cached to reduce API calls
4. Requires `OPENROUTER_API_KEY` environment variable for LLM fallback

## 3. DataConnector (Dynamic Loading)

**Location**: [charts/data_connector.py](charts/data_connector.py)

**Key Methods** ([data_connector.py:L1-L150](charts/data_connector.py#L1-L150)):

```python
load_all_datasets()                    # Loads all datasets from data/ subdirectories
switch_dataset(dataset_name)           # Changes active dataset + clears cached_data
get_available_datasets()               # Returns list of dataset names
get_current_dataset()                  # Returns active dataset name
extract_all_columns_info()             # Returns {table: {numeric: [], text: [], date: [], row_count}}
```

**Multi-Dataset Pattern**:

```python
data_connector.get_available_datasets()  # ['ecommerce', 'healthcare', 'finance']
data_connector.switch_dataset('healthcare')  # Switch active dataset
data_connector.get_current_dataset()  # Returns 'healthcare'
```

## 4. MetricCalculator (Calculated Metrics)

**Location**: [charts/metric_calculator.py](charts/metric_calculator.py)

**Supported Calculations**:

- `yoy_growth`: Year-over-year growth % → `((Current - Previous) / Previous) * 100`
- `mom_change`: Month-over-month change % → Similar formula for monthly data
- `percent_change`: Generic period-over-period % change
- `cumulative`: Running totals over time (cumsum)
- `moving_average`: N-period moving average (requires `calculation_window`)
- `per_unit`: Per-order/per-customer calculations (requires unit column)

**Usage Pattern**:

- ChartGenerator automatically invokes when `entities['calculation_type']` is present
- Handles both single-series and multi-series (grouped) data
- Adds calculated columns: `YoY_Growth_Pct`, `MoM_Change_Pct`, `Cumulative_Total`, `MA_{window}`

## 5. Dashboard Generation

**Location**: [dashboard/dashboard_generator.py](dashboard/dashboard_generator.py)

**ArchitectUI Themes** ([dashboard_generator.py:L26-L68](dashboard/dashboard_generator.py#L26-L68)):

```python
self.themes = {
    'modern': {...},  # Blue/purple gradient (default)
    'dark': {...},    # Dark mode
    'light': {...},   # Professional light
    'corporate': {...}
}
```

**Adding Charts**:

1. `add_chart_from_query(query, entities)` → appends to `self.charts[]`
2. `generate_dashboard()` → creates single HTML with all charts
3. Charts stored in `temp_dashboards/` (working) and `saved_dashboards/` (persistent)
