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
