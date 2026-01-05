# Gen-Dash: AI Coding Agent Instructions

## System Overview

Gen-Dash is an **AI-powered dashboard generation platform** that converts natural language queries into interactive data visualizations. The system uses a hybrid NLP approach (fuzzy matching + LLM fallback) to understand user intent and generate professional, role-based dashboards with AI-generated insights.

**Core Stack**: FastAPI (async backend), MySQL + SQLAlchemy, Plotly (charts), SmartQueryParser (NLP), LangChain RAG, Claude 3 Haiku (explanations)

**Current Version**: 1.2.0 (December 19, 2025)

## Architecture & Data Flow

### Request Pipeline

```
User Query → SmartQueryParser (fuzzy/LLM) → ChartGenerator → ArchitectUIDashboard → HTML/JSON
```

1. **NLU Layer** (`nlu/`): Query → Entities (metric, dimension, chart_type, aggregation)
2. **Chart Layer** (`charts/`): Entities + DataConnector → Plotly Figure
3. **Dashboard Layer** (`dashboard/`): Multiple Charts → Single HTML Dashboard
4. **API Layer** (`main.py`): FastAPI endpoints orchestrating the pipeline

### Critical Global State Pattern

**`dashboard_system` is initialized lazily on first use** (see [main.py](main.py#L450-L459)):

```python
dashboard_system = None  # Global singleton

def get_dashboard_system():
    global dashboard_system
    if dashboard_system is None:
        dashboard_system = InteractiveDashboard()
    return dashboard_system
```

- Charts accumulate in `dashboard_system.dashboard.charts[]` across requests
- Clear via `dashboard_system.dashboard.clear_charts()` before new sessions
- Avoid multiple `InteractiveDashboard()` instances - use `get_dashboard_system()`

## Key Components & Patterns

### 1. NLUChartPipeline (Query Processing)

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

### 2. SmartQueryParser (Hybrid NLP)

**Location**: [nlu/smart_query_parser.py](nlu/smart_query_parser.py)

**Pattern**: Fuzzy matching first (70-80% queries, <50ms), LLM fallback (20-30%, ~500ms)

```python
# Example: parse_query returns flat dict with lowercase keys
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

**Hybrid Approach** ([smart_query_parser.py:L47-L86](nlu/smart_query_parser.py#L47-L86)):

1. Fuzzy matching tries to match query terms to dataset columns (fast)
2. If confidence < 70 or ambiguous, falls back to LLM (Claude 3 Haiku via OpenRouter)
3. Results cached to reduce API calls
4. Requires `OPENROUTER_API_KEY` environment variable for LLM fallback

### 3. DataConnector Dynamic Loading

**Location**: [charts/data_connector.py](charts/data_connector.py)

**Multi-Dataset Architecture**:

- Auto-loads Excel files from `data/` folder OR waits for uploads (`auto_load` parameter)
- **Dataset Organization**: `data/{dataset_name}/` subdirectories
- Stores multiple datasets: `self.datasets = {dataset_name: {table_name: DataFrame}}`
- Active dataset tracked in `self.current_dataset`
- Cached active data in `self.cached_data` (dict of DataFrames)

**Key Methods** ([data_connector.py:L1-L150](charts/data_connector.py#L1-L150)):

```python
load_all_datasets()                    # Loads all datasets from data/ subdirectories
switch_dataset(dataset_name)           # Changes active dataset + clears cached_data
get_available_datasets()               # Returns list of dataset names
get_current_dataset()                  # Returns active dataset name
extract_all_columns_info()             # Returns {table: {numeric: [], text: [], date: [], row_count}}
```

**Column Detection Pattern**:

- Numeric columns: dtype check (int64, float64)
- Text/categorical: object dtype
- Date columns: datetime64 dtype or recognized date patterns
- **NO hardcoded datasets** - system adapts to any uploaded Excel structure

### 4. MetricCalculator (Calculated Metrics)

**Location**: [charts/metric_calculator.py](charts/metric_calculator.py)

**Supported Calculations** ([metric_calculator.py:L1-L100](charts/metric_calculator.py#L1-L100)):

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

**Multi-Series Support**: Detects `group_by_*` columns and calculates metrics per group

### 5. Role-Based Access Control (RBAC)

**Location**: [auth/auth.py](auth/auth.py)

**Session Management**:

- In-memory sessions dict: `sessions[session_id] = {user_id, role, department, ...}`
- 30-minute timeout: `SESSION_TIMEOUT = timedelta(minutes=30)`
- Decorators: `@Depends(require_auth)` and `require_role("admin")`

**Permission Matrix** ([auth.py:L18-L24](auth/auth.py#L18-L24)):

```python
ROLE_PERMISSIONS = {
    "Admin": ["admin", "analyst", "departmental", "viewer"],
    "Analyst": ["analyst", "departmental", "viewer"],
    "Departmental": ["departmental", "viewer"],
    "Viewer": ["viewer"]
}
```

### 6. Dashboard Generation Workflow

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

### 7. Dashboard Explainer (AI Insights)

**Location**: [dashboard/dashboard_explainer.py](dashboard/dashboard_explainer.py)

**v1.2.0 Major Update**: Replaced ALL hardcoded templates with LLM-generated explanations

**Two Operating Modes** ([dashboard_explainer.py:L1-L100](dashboard/dashboard_explainer.py#L1-L100)):

1. **Chart-Only Mode** (default when no documents provided):
   - AI analyzes chart data structure, metrics, dimensions
   - Generates comprehensive insights: trends, comparisons, actionable recommendations
   - Methods: `_explain_charts_only()`, `_generate_chart_only_explanation()`, `_extract_chart_insights()`
2. **RAG-Enhanced Mode** (when documents uploaded):
   - Combines chart analysis + company documents (PDFs/DOCX/PPTX)
   - Uses FAISS vector store for semantic search
   - Provides context-aware insights linked to business documents

**Key Methods**:

```python
explain_dashboard(queries, force_chart_only=False)  # Main entry point
load_company_profile(file_path)                    # Load docs for RAG
explain_single_chart(query, entities)              # Single chart explanation
get_comparative_insights(queries_list)             # Multi-chart comparison
```

**LLM Configuration**:

- Model: Claude 3 Haiku (via OpenRouter)
- Temperature: 0.3 (focused but slightly creative)
- Max tokens: 800
- Requires: `OPENROUTER_API_KEY` environment variable

**API Endpoints**:

- `/api/explain-dashboard` POST: `{queries: [], force_chart_only: bool}`
- `/api/load-company-profile` POST: Upload PDF/DOCX/PPTX for RAG context

## Development Workflows

### Logging System (v1.1.0)

**Structured Logging** ([main.py:L21-L30](main.py#L21-L30)):

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),                    # Console output
        logging.FileHandler('logs/gendash.log', mode='a')  # File output
    ]
)
```

**Module-specific loggers**:

- Use `logger = logging.getLogger(__name__)` in each module
- Logs written to `logs/gendash.log` (rotating, 10MB max, 5 backups)
- Available modules: app, database, auth, charts, dashboard, nlu, rag

**Error Handling Pattern**: All API endpoints use structured error responses:

```python
try:
    # Business logic
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
finally:
    # Cleanup (e.g., db.close() handled by get_db() dependency)
```

### Environment Setup

**Python Virtual Environment** (required):

```bash
# Create venv (first time only)
python3 -m venv venv

# Activate venv (every session)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# 1. Activate virtual environment (REQUIRED - do this FIRST in every terminal session):
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 2. Start the FastAPI server from project root:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access**: http://localhost:8000

**Critical**: The virtual environment MUST be activated before running any Python commands or starting the server. Check for `(venv)` prefix in terminal prompt.

### Database Setup

**Connection String**: `mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard`

**Connection Pooling** ([main.py:L65-L71](main.py#L65-L71)):

- Pool size: 5 connections
- Max overflow: 10 connections
- Connection recycling: Every 3600 seconds (1 hour)
- Health checks enabled: `pool_pre_ping=True`

```python
# Initialize tables (from project root with venv activated):
python -c "from database.models import Base; from main import engine; Base.metadata.create_all(bind=engine)"
```

**Models** ([database/models.py](database/models.py)):

- `User`: id, username, email, hashed_password, role, department
- `Dashboard`: id, user_id, title, charts_config (JSON), file_path, visible_to_viewer, allowed_departments

### Testing Query Pipeline

```bash
# Standalone NLU test (no server required - with venv activated):
python3 nlu/chart_pipeline.py "sales by region"
python3 nlu/chart_pipeline.py "average revenue by product"

# Test NER model (if using custom NER instead of SmartQueryParser):
python3 nlu/test_ner.py
```

**Note**: Current system uses SmartQueryParser (fuzzy + LLM hybrid), not the trained NER model by default.

### Data Upload & Management

**Upload Endpoint**: `/api/upload-data` (POST with `UploadFile`)

**Workflow**:

1. User uploads Excel files via `UploadFile` (no auto_load)
2. Files saved to `data/{dataset_name}/` directory
3. DataConnector reloads with `auto_load=False` → only uploaded files
4. System adapts to new columns automatically
5. Use `/api/get-dataset-info` to fetch schema after upload

**Multi-Dataset Pattern**:

```python
data_connector.get_available_datasets()  # ['ecommerce', 'healthcare', 'finance']
data_connector.switch_dataset('healthcare')  # Switch active dataset
data_connector.get_current_dataset()  # Returns 'healthcare'
```

### RAG System (Optional)

**Location**: [rag/rag.py](rag/rag.py)

- Requires `OPENROUTER_API_KEY` in `.env`
- Multi-format loader: PDF, DOCX, PPTX via [rag/loaders.py](rag/loaders.py)
- FAISS vector store for semantic search
- Used by dashboard explainer for AI-generated insights in RAG-enhanced mode

## Project-Specific Conventions

### 1. Import Patterns

**Dashboard modules must add parent to path**:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from charts.chart_generator import ChartGenerator
```

### 2. Logging Setup

**Standard pattern** ([main.py:L21-L30](main.py#L21-L30)):

```python
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/gendash.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)
```

### 3. Error Handling in API Routes

**Always return structured JSON errors**:

```python
try:
    # Business logic
except Exception as e:
    logger.error(f"Error: {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
```

### 4. File Paths

- **Exports**: `exports/dashboard_{id}_data_{timestamp}_chart{n}.csv`
- **Saved Dashboards**: `saved_dashboards/dashboard_{id}_{title}_{timestamp}.html`
- **Temp Dashboards**: `temp_dashboards/dashboard_snapshot_{id}_{timestamp}.html`

### 5. Frontend Serving Pattern

**Static HTML from `Frontend/` folder**:

```python
@app.get("/admin_dashboard.html", response_class=HTMLResponse)
async def admin_dashboard(session: dict = Depends(require_auth)):
    file_path = Path("Frontend/admin_dashboard.html")
    return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
```

## Common Pitfalls

1. **Don't instantiate multiple `InteractiveDashboard()` instances** - use `get_dashboard_system()`
2. **Clear charts between sessions** - charts accumulate globally: `dashboard_system.dashboard.clear_charts()`
3. **SmartQueryParser entity keys are lowercase** - `metric` not `METRIC`, and strings not lists
4. **DataConnector requires real data** - no fallback to synthetic data; upload to `data/` folder
5. **Session cookies named `session_id`** - check `Cookie(None)` in FastAPI deps
6. **Database connection pooling** - use `get_db()` dependency, never create sessions manually
7. **Virtual environment must be activated** - run `source venv/bin/activate` before starting server
8. **Dashboard Explainer requires API key** - set `OPENROUTER_API_KEY` in `.env` for AI insights
9. **File uploads use UploadFile type** - FastAPI endpoint: `file: UploadFile = File(...)`
10. **Calculated metrics need time columns** - YoY/MoM require date dimension in data

## Key Files Reference

| File                                                                 | Purpose                 | Critical Patterns                         |
| -------------------------------------------------------------------- | ----------------------- | ----------------------------------------- |
| [main.py](main.py)                                                   | FastAPI app entry point | Lifespan events, `get_dashboard_system()` |
| [nlu/smart_query_parser.py](nlu/smart_query_parser.py)               | Query → Entities        | Fuzzy + LLM hybrid, entity dict format    |
| [charts/chart_generator.py](charts/chart_generator.py)               | Entities → Plotly chart | Dynamic data loading, no hardcoding       |
| [charts/metric_calculator.py](charts/metric_calculator.py)           | Calculated metrics      | YoY, MoM, cumulative, moving average      |
| [dashboard/dashboard_generator.py](dashboard/dashboard_generator.py) | Charts → HTML dashboard | ArchitectUI themes, multi-chart layout    |
| [dashboard/dashboard_explainer.py](dashboard/dashboard_explainer.py) | AI insights generation  | Claude 3 Haiku, chart + RAG modes         |
| [auth/auth.py](auth/auth.py)                                         | Session & RBAC          | In-memory sessions, permission matrix     |
| [database/models.py](database/models.py)                             | SQLAlchemy ORM          | User, Dashboard tables                    |

## Environment Variables

Required in `.env`:

```bash
DATABASE_URL=mysql+pymysql://user:pass@localhost/analytics_dashboard
OPENROUTER_API_KEY=sk-or-...  # For SmartQueryParser LLM + Dashboard Explainer (Claude 3 Haiku)
SECRET_KEY=...                 # For JWT (if using tokens)
```

**Note**: As of v1.2.0, `OPENROUTER_API_KEY` is **required** for dashboard explanations (no hardcoded fallbacks)

## When Making Changes

- **Adding new chart types**: Update `chart_keywords` in [SmartQueryParser](nlu/smart_query_parser.py#L71-L79)
- **New themes**: Add to `self.themes` dict in [ArchitectUIDashboard](dashboard/dashboard_generator.py#L26-L68)
- **New roles**: Update `ROLE_PERMISSIONS` in [auth.py](auth/auth.py#L18-L24) and database schema
- **Data formats**: Extend [loaders.py](rag/loaders.py) for new file types
- **API endpoints**: Follow session-checking pattern with `Depends(require_auth)`

---

**Last Updated**: January 5, 2026 | **Version**: 1.2.0
