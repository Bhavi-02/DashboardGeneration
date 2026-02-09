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
