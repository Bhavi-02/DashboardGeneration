# Gen-Dash Development Guide

## Environment Setup

### Python Virtual Environment (Required)

```bash
# Create venv (first time only)
python3 -m venv venv

# Activate venv (every session - REQUIRED)
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Critical**: The virtual environment MUST be activated before running any Python commands or starting the server. Check for `(venv)` prefix in terminal prompt.

## Running the Application

```bash
# 1. Activate virtual environment (do this FIRST):
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 2. Start the FastAPI server from project root:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access**: http://localhost:8000

## Database Setup

### Connection Configuration

**Connection String**: `mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard`

**Connection Pooling** ([main.py:L65-L71](main.py#L65-L71)):

- Pool size: 5 connections
- Max overflow: 10 connections
- Connection recycling: Every 3600 seconds (1 hour)
- Health checks enabled: `pool_pre_ping=True`

### Initialize Database Tables

```bash
# From project root with venv activated:
python -c "from database.models import Base; from main import engine; Base.metadata.create_all(bind=engine)"
```

## Logging System

### Configuration

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

### Module-Specific Loggers

Use `logger = logging.getLogger(__name__)` in each module.

**Available Modules**: app, database, auth, charts, dashboard, nlu, rag

**Log Files**:

- Location: `logs/gendash.log`
- Rotation: 10MB max, 5 backups

## Testing

### Test Query Pipeline

```bash
# Standalone NLU test (no server required - with venv activated):
python3 nlu/chart_pipeline.py "sales by region"
python3 nlu/chart_pipeline.py "average revenue by product"

# Test NER model (if using custom NER instead of SmartQueryParser):
python3 nlu/test_ner.py
```

**Note**: Current system uses SmartQueryParser (fuzzy + LLM hybrid), not the trained NER model by default.

## Data Management Workflow

### Upload & Management

**Upload Endpoint**: `/api/upload-data` (POST with `UploadFile`)

**Workflow**:

1. User uploads Excel files via `UploadFile` (no auto_load)
2. Files saved to `data/{dataset_name}/` directory
3. DataConnector reloads with `auto_load=False` → only uploaded files
4. System adapts to new columns automatically
5. Use `/api/get-dataset-info` to fetch schema after upload

### Multi-Dataset Pattern

```python
data_connector.get_available_datasets()  # ['ecommerce', 'healthcare', 'finance']
data_connector.switch_dataset('healthcare')  # Switch active dataset
data_connector.get_current_dataset()  # Returns 'healthcare'
```

## RAG System (Optional)

**Location**: [rag/rag.py](rag/rag.py)

- Requires `OPENROUTER_API_KEY` in `.env`
- Multi-format loader: PDF, DOCX, PPTX via [rag/loaders.py](rag/loaders.py)
- FAISS vector store for semantic search
- Used by dashboard explainer for AI-generated insights in RAG-enhanced mode

## Environment Variables

Create `.env` file in project root:

```bash
DATABASE_URL=mysql+pymysql://user:pass@localhost/analytics_dashboard
OPENROUTER_API_KEY=sk-or-...  # For SmartQueryParser LLM + Dashboard Explainer (Claude 3 Haiku)
SECRET_KEY=...                 # For JWT (if using tokens)
```

**Note**: As of v1.2.0, `OPENROUTER_API_KEY` is **required** for dashboard explanations (no hardcoded fallbacks)

## Key Files

| File                                                                 | Purpose                  |
| -------------------------------------------------------------------- | ------------------------ |
| [main.py](main.py)                                                   | FastAPI app entry point  |
| [main.py:L38-L180](main.py#L38-L180)                                 | Session & rating systems |
| [nlu/smart_query_parser.py](nlu/smart_query_parser.py)               | Query → Entities         |
| [charts/chart_generator.py](charts/chart_generator.py)               | Entities → Plotly chart  |
| [charts/data_connector.py](charts/data_connector.py)                 | Multi-dataset management |
| [charts/metric_calculator.py](charts/metric_calculator.py)           | Calculated metrics       |
| [dashboard/dashboard_generator.py](dashboard/dashboard_generator.py) | Charts → HTML dashboard  |
| [dashboard/smart_generator.py](dashboard/smart_generator.py)         | AI chart recommendations |
| [dashboard/dashboard_explainer.py](dashboard/dashboard_explainer.py) | AI insights generation   |
| [auth/auth.py](auth/auth.py)                                         | Session & RBAC           |
| [database/models.py](database/models.py)                             | SQLAlchemy ORM           |
