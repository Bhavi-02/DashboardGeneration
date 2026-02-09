# Gen-Dash API Reference

## Dashboard Generation

### Generate Dashboard from Queries

`POST /api/generate-dashboard`

Generate dashboard from multiple natural language queries.

**Request Body**:

```json
{
  "queries": ["sales by region", "revenue over time"],
  "theme": "modern"
}
```

### AI-Powered Smart Dashboard

`POST /api/generate-smart-dashboard`

Automatic dashboard generation using AI recommendations.

**Request Body**:

```json
{
  "num_charts": 5,
  "custom_prompt": "focus on CEO metrics",
  "override_context": {
    "department": "Sales",
    "role": "Manager"
  }
}
```

**Notes**:

- Clears existing charts before generation
- Requires `OPENROUTER_API_KEY` for LLM
- Supports session-based personalization

## User Feedback & Personalization

### Rate Chart

`POST /api/rate-chart`

Unified rating endpoint supporting 3-state ratings: like/neutral/dislike.

**Request Body**:

```json
{
  "chart": {
    "signature": "sales|region|bar|sum||",
    "metric": "sales",
    "dimension": "region",
    "chart_type": "bar",
    "title": "Sales by Branch",
    "reasoning": "..."
  },
  "rating": "like"
}
```

**Notes**:

- Updates persistent time-decay rating file (`data/user_ratings/user_{id}_ratings.json`)
- Uses v2 schema with rating_history
- Session feedback automatically tracked in RAM
- Ephemeral session data cleared on logout

## Data Management

### Upload Data

`POST /api/upload-data`

Upload Excel files to data folder.

**Request**: Multipart form with `UploadFile`

**Notes**:

- Multi-dataset support: saves to `data/{dataset_name}/`
- Auto-reloads DataConnector
- System adapts to new columns automatically

### Get Dataset Info

`GET /api/get-dataset-info`

Get current dataset schema and available datasets.

**Response**:

```json
{
  "current_dataset": "ecommerce",
  "available_datasets": ["ecommerce", "healthcare", "finance"],
  "tables": {
    "sales": {
      "numeric": ["amount", "quantity"],
      "text": ["region", "product"],
      "date": ["date"],
      "row_count": 1000
    }
  }
}
```

### Switch Dataset

`POST /api/switch-dataset`

Switch active dataset.

**Request Body**:

```json
{
  "dataset_name": "healthcare"
}
```

## AI Explanations

### Explain Dashboard

`POST /api/explain-dashboard`

Get AI-generated insights for charts.

**Request Body**:

```json
{
  "queries": ["sales by region", "revenue over time"],
  "force_chart_only": false
}
```

**Modes**:

- **Chart-only mode** (`force_chart_only: true`): Analyzes charts only
- **RAG-enhanced mode** (default): Uses uploaded documents for context-aware insights

### Load Company Profile

`POST /api/load-company-profile`

Upload PDF/DOCX/PPTX for RAG context.

**Request**: Multipart form with document file

## Error Handling Pattern

All endpoints return structured JSON errors:

```python
try:
    # Business logic
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return JSONResponse({"error": str(e)}, status_code=500)
```

## Authentication

All endpoints require authentication via `@Depends(require_auth)`.

**Session Management**:

- Session cookie: `session_id`
- 30-minute timeout
- Check `Cookie(None)` in FastAPI dependencies
