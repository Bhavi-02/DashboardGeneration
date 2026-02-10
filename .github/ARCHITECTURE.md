# Gen-Dash Architecture

## System Overview

Gen-Dash is an **AI-powered dashboard generation platform** that converts natural language queries into interactive data visualizations.

**Core Stack**: FastAPI (async backend), MySQL + SQLAlchemy, Plotly (charts), SmartQueryParser (NLP), LangChain RAG, Claude 3 Haiku (explanations + smart generation)

## Request Pipeline

```
User Query → SmartQueryParser (fuzzy/LLM) → ChartGenerator → ArchitectUIDashboard → HTML/JSON
```

### Pipeline Layers

1. **NLU Layer** (`nlu/`): Query → Entities (metric, dimension, chart_type, aggregation)
2. **Chart Layer** (`charts/`): Entities + DataConnector → Plotly Figure
3. **Dashboard Layer** (`dashboard/`): Multiple Charts → Single HTML Dashboard
4. **API Layer** (`main.py`): FastAPI endpoints orchestrating the pipeline

## Critical Global State Pattern

**`dashboard_system` is initialized lazily on first use** (see [main.py](main.py#L450-L459)):

```python
dashboard_system = None  # Global singleton

def get_dashboard_system():
    global dashboard_system
    if dashboard_system is None:
        dashboard_system = InteractiveDashboard()
    return dashboard_system
```

**Key Points**:

- Charts accumulate in `dashboard_system.dashboard.charts[]` across requests
- Clear via `dashboard_system.dashboard.clear_charts()` before new sessions
- Avoid multiple `InteractiveDashboard()` instances - use `get_dashboard_system()`

## Multi-Dataset Architecture

**Location**: [charts/data_connector.py](charts/data_connector.py)

- Auto-loads Excel files from `data/` folder OR waits for uploads (`auto_load` parameter)
- **Dataset Organization**: `data/{dataset_name}/` subdirectories
- Stores multiple datasets: `self.datasets = {dataset_name: {table_name: DataFrame}}`
- Active dataset tracked in `self.current_dataset`
- Cached active data in `self.cached_data` (dict of DataFrames)

**Column Detection Pattern**:

- Numeric columns: dtype check (int64, float64)
- Text/categorical: object dtype
- Date columns: datetime64 dtype or recognized date patterns
- **NO hardcoded datasets** - system adapts to any uploaded Excel structure

## Role-Based Access Control (RBAC)

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

## Database Architecture

**Connection String**: `mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard`

**Connection Pooling** ([main.py:L65-L71](main.py#L65-L71)):

- Pool size: 5 connections
- Max overflow: 10 connections
- Connection recycling: Every 3600 seconds (1 hour)
- Health checks enabled: `pool_pre_ping=True`

**Models** ([database/models.py](database/models.py)):

- `User`: id, username, email, hashed_password, role, department
- `Dashboard`: id, user_id, title, charts_config (JSON), file_path, visible_to_viewer, allowed_departments

## Tech Stack Summary

| Component       | Technology                                         |
| --------------- | -------------------------------------------------- |
| Backend         | FastAPI (async), uvicorn                           |
| Database        | MySQL + SQLAlchemy ORM (pool_size=5, recycle=3600) |
| Charts          | Plotly (interactive)                               |
| NLP Query       | SmartQueryParser (fuzzy + LLM hybrid)              |
| AI/LLM          | Claude 3 Haiku via OpenRouter                      |
| RAG             | LangChain + FAISS (vector similarity)              |
| Auth            | Bcrypt + Session cookies (30-min timeout)          |
| Personalization | Session store (RAM) + Time-decay (JSON files)      |

## Request Pipeline Detailed

```
HTTP Request (session_id cookie)
    ↓
@Depends(require_auth) → Session validation + RBAC check
    ↓
@Depends(get_dashboard_system) → Get singleton instance
    ↓
@Depends(get_db) → SQLAlchemy session (connection pool)
    ↓
Router handler → Business logic in services/
    ↓
JSONResponse or HTMLResponse
```

## Chart Generation Flow

```
User Query String
    ↓
SmartQueryParser (fuzzy 70-80% success, LLM fallback 20-30%)
    ↓
entities dict (lowercase keys, flat structure)
    ↓
ChartGenerator.generate_chart(entities, query)
    ↓
MetricCalculator (if calculation_type specified)
    ↓
Plotly figure generation
    ↓
DashboardGenerator.add_chart_from_query()
    ↓
Accumulates in dashboard_system.dashboard.charts[]
```

## Key File Locations (main.py:1-95)

```python
# Entry point - minimal orchestration only
/main.py (95 lines)

# Core infrastructure
/core/config.py      # Logging, CORS
/core/database.py    # SQLAlchemy pool
/core/lifespan.py    # Startup/shutdown

# Business logic
/services/dashboard_service.py  # Singleton management
/services/auth_service.py       # Password hashing
/services/feedback_service.py   # Personalization

# API layer (57 endpoints across 9 routers)
/api/routers/auth.py            # Session management
/api/routers/dashboards.py      # Dashboard CRUD + RBAC
/api/routers/charts.py          # Chart generation
/api/routers/datasets.py        # Data switching
/api/routers/ratings.py         # User feedback
/api/routers/explainability.py  # AI insights
/api/routers/rag.py             # Document Q&A

# AI/ML features
/dashboard/smart_generator.py   # 4-component agentic system
/nlu/smart_query_parser.py      # Hybrid NLP (fuzzy + LLM)
/dashboard/dashboard_explainer.py # AI insights generator

# Data layer
/charts/data_connector.py       # Multi-dataset loader
/charts/chart_generator.py      # Plotly chart creation
/charts/metric_calculator.py    # YoY, MoM, MA calculations
```
