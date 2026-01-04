# Gen-Dash: AI Coding Agent Instructions

## System Overview

Gen-Dash is an **AI-powered dashboard generation platform** that converts natural language queries into interactive data visualizations. The system uses a hybrid NLP approach (fuzzy matching + LLM fallback) to understand user intent and generate professional, role-based dashboards.

**Core Stack**: FastAPI (async backend), MySQL + SQLAlchemy, Plotly (charts), SmartQueryParser (NLP), LangChain RAG

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

### 1. SmartQueryParser (Hybrid NLP)

**Location**: [nlu/smart_query_parser.py](nlu/smart_query_parser.py)

**Pattern**: Fuzzy matching first (70-80% queries, <50ms), LLM fallback (20-30%, ~500ms)

```python
# Example: parse_query returns flat dict (NOT uppercase NER keys)
entities = {
    'metric': 'sales',           # NOT 'METRIC': ['sales']
    'dimension': 'region',
    'chart_type': 'bar',
    'aggregation': 'sum',
    'filters': []
}
```

**Migration Note**: Old NER format used uppercase keys + lists. Auto-migration at [smart_query_parser.py:L417](nlu/smart_query_parser.py#L417). See [docs/MIGRATION_SMART_PARSER.md](docs/MIGRATION_SMART_PARSER.md).

### 2. DataConnector Dynamic Loading

**Location**: [charts/data_connector.py](charts/data_connector.py)

- Auto-loads Excel/CSV from `data/` folder on init
- Cached in `self.cached_data` (dict of DataFrames)
- Column extraction: `extract_all_columns_info()` returns `{table_name: {numeric: [], categorical: []}}`
- **NO hardcoded datasets** - system adapts to any data structure

### 3. Role-Based Access Control (RBAC)

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

### 4. Dashboard Generation Workflow

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

## Development Workflows

### Running the Application

```bash
# Terminal from project root:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access**: http://localhost:8000

### Database Setup

**Connection String**: `mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard`

```python
# Initialize tables (from project root):
python -c "from database.models import Base; from main import engine; Base.metadata.create_all(bind=engine)"
```

**Models** ([database/models.py](database/models.py)):

- `User`: id, username, email, hashed_password, role, department
- `Dashboard`: id, user_id, title, charts_config (JSON), file_path, visible_to_viewer, allowed_departments

### Testing Query Pipeline

```bash
# Standalone NLU test (no server):
python3 nlu/chart_pipeline.py "sales by region"
python3 nlu/chart_pipeline.py "average revenue by product"
```

### RAG System (Optional)

**Location**: [rag/rag.py](rag/rag.py)

- Requires `OPENROUTER_API_KEY` in `.env`
- Multi-format loader: PDF, DOCX, PPTX via [rag/loaders.py](rag/loaders.py)
- FAISS vector store for semantic search
- Used by dashboard explainer for AI-generated insights

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

## Key Files Reference

| File                                                                 | Purpose                 | Critical Patterns                         |
| -------------------------------------------------------------------- | ----------------------- | ----------------------------------------- |
| [main.py](main.py)                                                   | FastAPI app entry point | Lifespan events, `get_dashboard_system()` |
| [nlu/smart_query_parser.py](nlu/smart_query_parser.py)               | Query → Entities        | Fuzzy + LLM hybrid, entity dict format    |
| [charts/chart_generator.py](charts/chart_generator.py)               | Entities → Plotly chart | Dynamic data loading, no hardcoding       |
| [dashboard/dashboard_generator.py](dashboard/dashboard_generator.py) | Charts → HTML dashboard | ArchitectUI themes, multi-chart layout    |
| [auth/auth.py](auth/auth.py)                                         | Session & RBAC          | In-memory sessions, permission matrix     |
| [database/models.py](database/models.py)                             | SQLAlchemy ORM          | User, Dashboard tables                    |

## Environment Variables

Required in `.env`:

```bash
DATABASE_URL=mysql+pymysql://user:pass@localhost/analytics_dashboard
OPENROUTER_API_KEY=sk-...  # For SmartQueryParser LLM fallback
SECRET_KEY=...              # For JWT (if using tokens)
```

## When Making Changes

- **Adding new chart types**: Update `chart_keywords` in [SmartQueryParser](nlu/smart_query_parser.py#L71-L79)
- **New themes**: Add to `self.themes` dict in [ArchitectUIDashboard](dashboard/dashboard_generator.py#L26-L68)
- **New roles**: Update `ROLE_PERMISSIONS` in [auth.py](auth/auth.py#L18-L24) and database schema
- **Data formats**: Extend [loaders.py](rag/loaders.py) for new file types
- **API endpoints**: Follow session-checking pattern with `Depends(require_auth)`

---

**Last Updated**: January 4, 2026 | **Version**: 1.1.0
