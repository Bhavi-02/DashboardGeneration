# Gen-Dash Conventions & Best Practices

## Import Patterns

### Dashboard Module Imports

Dashboard modules must add parent to path:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from charts.data_connector import DataConnector
from charts.chart_generator import ChartGenerator
```

### Standard Logging Pattern

```python
import logging

logger = logging.getLogger(__name__)

# Usage
logger.info("Operation completed")
logger.error(f"Error occurred: {error}")
```

## Error Handling in API Routes

Always return structured JSON errors:

```python
try:
    # Business logic
except Exception as e:
    logger.error(f"Error: {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
```

## File Path Conventions

Use absolute paths for serving HTML files:

```python
@app.get("/admin_dashboard.html", response_class=HTMLResponse)
async def admin_dashboard(session: dict = Depends(require_auth)):
    file_path = Path("Frontend/admin_dashboard.html")
    return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
```

## Common Pitfalls

### Global State Management

1. **Don't instantiate multiple `InteractiveDashboard()` instances** - use `get_dashboard_system()`
2. **Clear charts between sessions** - charts accumulate globally: `dashboard_system.dashboard.clear_charts()`

### Data & Query Processing

3. **SmartQueryParser entity keys are lowercase** - `metric` not `METRIC`, and strings not lists
4. **DataConnector requires real data** - no fallback to synthetic data; upload to `data/` folder
5. **Calculated metrics need time columns** - YoY/MoM require date dimension in data

### Authentication & Sessions

6. **Session cookies named `session_id`** - check `Cookie(None)` in FastAPI deps
7. **Database connection pooling** - use `get_db()` dependency, never create sessions manually

### Development Environment

8. **Virtual environment must be activated** - run `source venv/bin/activate` before starting server
9. **Dashboard Explainer requires API key** - set `OPENROUTER_API_KEY` in `.env` for AI insights

### File Operations

10. **File uploads use UploadFile type** - FastAPI endpoint: `file: UploadFile = File(...)`

### AI Features

11. **Smart Generator uses structured_output** - Pydantic models define LLM response schema
12. **Smart Generator clears existing charts** - Always clears dashboard before auto-generation

### Personalization Systems

13. **Session feedback vs. time-decay ratings** - Session feedback is ephemeral (RAM), ratings are persistent (JSON files)
14. **Rating file schema version** - Current version is v2 with unified rating structure; check `version` field
15. **Time-decay weights are automatic** - Don't manually calculate; use `_calculate_decay_weight()` function

### Multi-Dataset Operations

16. **Multi-dataset switching** - Always call `switch_dataset()` before accessing data; use `get_current_dataset()` to verify active dataset

## When Making Changes

### Adding New Features

- **New chart types**: Update `chart_keywords` in [SmartQueryParser](nlu/smart_query_parser.py#L71-L79) AND update `SmartChartRecommender` LLM prompt
- **New themes**: Add to `self.themes` dict in [ArchitectUIDashboard](dashboard/dashboard_generator.py#L26-L68)
- **New roles**: Update `ROLE_PERMISSIONS` in [auth.py](auth/auth.py#L18-L24) and database schema
- **Data formats**: Extend [loaders.py](rag/loaders.py) for new file types

### API & Integration

- **API endpoints**: Follow session-checking pattern with `Depends(require_auth)`
- **Department contexts**: Update `ContextAnalyzer.build_user_context()` in [smart_generator.py](dashboard/smart_generator.py) for new departments
- **LLM prompts**: Both SmartGenerator and DashboardExplainer use ChatOpenAI with structured output - modify Pydantic models for schema changes

### Data & Personalization

- **Rating system modifications**: Update both session feedback store logic AND persistent rating file schema; maintain v2 format compatibility
- **Multi-dataset support**: When adding new datasets, create subdirectories in `data/` folder; use `get_available_datasets()` to verify loading

## Code Style Guidelines

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Type Hints

Use type hints for function parameters and return values:

```python
def process_query(query: str, use_llm: bool = True) -> Dict[str, Any]:
    """Process natural language query into entities"""
    ...
```

### Docstrings

Use docstrings for classes and public methods:

```python
def generate_chart(self, entities: Dict[str, Any], query: str) -> Figure:
    """
    Generate Plotly chart from entities

    Args:
        entities: Parsed query entities
        query: Original natural language query

    Returns:
        Plotly Figure object
    """
    ...
```

## Critical Patterns (Must Know)

### 1. Global Dashboard System Singleton ⚠️ CRITICAL

```python
# ✅ ALWAYS use this pattern - import from services
from services.dashboard_service import get_dashboard_system
dashboard_system = get_dashboard_system()

# ❌ NEVER instantiate directly - causes chart accumulation bugs
from dashboard.interactive_dashboard import InteractiveDashboard
dashboard_system = InteractiveDashboard()  # WRONG - creates duplicate instance
```

**Why this matters**: Charts accumulate in `dashboard_system.dashboard.charts[]` across requests. Multiple instances = lost charts.

**Chart lifecycle pattern**:

```python
# 1. Get singleton
dashboard_system = get_dashboard_system()

# 2. Clear previous session charts
dashboard_system.dashboard.clear_charts()

# 3. Add charts
dashboard_system.dashboard.add_chart_from_query(query, entities)

# 4. Generate final dashboard
dashboard_system.dashboard.generate_and_save_dashboard(user_id, title)
```

### 2. Entity Structure - Lowercase Keys Required ⚠️

```python
# ✅ CORRECT - flat dict with lowercase string keys
entities = {
    'metric': 'sales',               # String value, not list
    'dimension': 'region',           # Lowercase key
    'chart_type': 'bar',            # bar, line, pie, area, scatter, heatmap, radar
    'aggregation': 'sum',           # sum, avg, count, min, max
    'filters': [],                  # List of filter conditions
    'group_by': 'category',         # Optional secondary dimension
    'time_granularity': 'monthly',  # daily, monthly, quarterly, yearly
    'calculation_type': 'yoy_growth',  # Optional: yoy_growth, mom_change, etc.
}

# ❌ WRONG - common mistakes
entities = {
    'METRIC': 'sales',              # Uppercase keys not recognized
    'metric': ['sales', 'profit'],  # Lists not supported for entity values
    'Dimension': 'region',          # Mixed case fails
}
```

**Used in**: `ChartGenerator.generate_chart(entities, query)`, `SmartQueryParser.parse_query()`, `ChartRecommendation` model

### 3. Agentic AI System - 4-Component Architecture

**Location**: `dashboard/smart_generator.py` (1254 lines)

```python
from dashboard.smart_generator import SmartDashboardGenerator

# Orchestrates 4 components automatically:
generator = SmartDashboardGenerator(data_connector, use_llm=True)
result = generator.generate_smart_dashboard(
    user_department="Finance",
    user_role="Analyst",
    num_charts=5,
    custom_prompt="focus on CEO-level KPIs"  # Optional override
)
# Returns: {success, recommendations, charts, profile, context, failed_recommendations}
```

**4 Components** (orchestrated automatically):

1. **DataProfiler** → Analyzes schema, detects industry, identifies semantic columns
2. **ContextAnalyzer** → Maps department → preferred metrics, role → chart types
3. **SmartChartRecommender** → LLM-powered (Claude 3 Haiku), validates columns with multi-layer fallback
4. **SmartDashboardGenerator** → Orchestrates workflow, generates charts, handles failures

### 4. Dual Personalization System

**Tier 1: Session Feedback** (Ephemeral, RAM-based)

- `services/feedback_service.py` → `session_feedback_store[session_id]`
- Tracks likes/dislikes during active session
- Cleared on logout/timeout
- Used for immediate session personalization

**Tier 2: Time-Decay Ratings** (Persistent, JSON file-based)

- File: `data/user_ratings/user_{id}_ratings.json` (v2 schema)
- Exponential decay: `weight = e^(-0.02 × days_old)` (35-day half-life)
- Survives across sessions
- Used for long-term user preference modeling

### 5. Multi-Dataset Architecture

```python
# DataConnector maintains single active dataset
data_connector.get_available_datasets()     # List all in data/ subdirs
data_connector.switch_dataset('healthcare') # Switch active dataset
data_connector.get_current_dataset()        # Returns 'healthcare'

# Auto-detection (NO hardcoded column names):
# - Numeric: int64, float64 dtypes
# - Categorical: object dtype
# - Date: datetime64 or recognized patterns
```

**File structure**: `data/{dataset_name}/{tables}.xlsx`
