# Gen-Dash: Intelligent Dashboard Generation Platform

A comprehensive, AI-powered dashboard generation and visualization platform built with FastAPI, featuring advanced NLP capabilities, role-based access control, and interactive data exploration tools.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Role-Based Access Control](#role-based-access-control)
- [Advanced Features](#advanced-features)
- [Data Management](#data-management)
- [Development](#development)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## Overview

Gen-Dash is an enterprise-grade dashboard generation platform that leverages natural language processing, retrieval-augmented generation, and machine learning to enable non-technical users to create professional, interactive data visualizations from complex datasets.

<<<<<<< HEAD
**Current Version:** 1.0.0  
**Build Status:** Active Development  
**Last Updated:** December 2025
=======
**Current Version:** 1.1.0  
**Build Status:** Production Ready  
**Last Updated:** December 19, 2025

> > > > > > > chart-creator

## Key Features

### Authentication & Authorization

- **Secure User Management**: Bcrypt-hashed passwords with session-based authentication
- **Role-Based Access Control**: Admin, Analyst, Viewer, and Department Head roles
- **Session Management**: Secure token-based session handling with JWT support
- **Department-Based Visibility**: Restrict dashboard access by department

### Dashboard & Visualization

- **Interactive Dashboards**: Real-time, responsive Plotly-based visualizations
- **Multiple Chart Types**: Area, bar, line, pie, radar, and custom chart support
- **Theme Support**: Modern, dark, light, and corporate theme options
- **Export Capabilities**: Save dashboards as HTML/interactive formats
- **ArchitectUI Design**: Professional, modern UI with smooth animations

### Natural Language Processing

- **Named Entity Recognition (NER)**: Extract business entities from text
- **Intent Classification**: Identify user intent from natural language queries
- **Trained Models**: DistilBERT-based models for accurate NLP predictions
- **Custom Pipeline**: Specialized NLU pipeline for business intelligence

### Retrieval-Augmented Generation (RAG)

- **Multi-Format Document Support**: PDF, DOCX, PPTX, and more
- **Intelligent Document Loading**: Automatic format detection and extraction
- **Vector-Based Search**: FAISS integration for semantic similarity search
- **Context-Aware Responses**: LangChain integration for intelligent Q&A

### Data Management

- **MySQL Database**: Persistent storage with SQLAlchemy ORM
- **Data Export**: CSV exports of dashboard data with timestamps
- **Training Data**: Support for annotated and raw training datasets
- **Data Versioning**: Track dashboard snapshots over time

### User Experience

- **Multi-Page Frontend**: Dedicated pages for different user roles
- **Admin Dashboard**: Comprehensive system administration interface
- **Analyst Tools**: Advanced data exploration and chart creation
- **Viewer Portal**: Read-only access to shared dashboards

## Architecture

### Technology Stack

| Layer                 | Technology                           |
| --------------------- | ------------------------------------ |
| **Backend Framework** | FastAPI (async Python web framework) |
| **Database**          | MySQL with SQLAlchemy ORM            |
| **Authentication**    | Bcrypt + JWT + Session Management    |
| **Visualization**     | Plotly (interactive charts)          |
| **NLP/ML**            | Transformers, Torch, scikit-learn    |
| **Search**            | FAISS (Vector similarity)            |
| **Frontend**          | HTML5, CSS3, JavaScript              |
| **Containerization**  | Python venv (virtual environment)    |

### System Architecture

```
Gen-Dash
├── Frontend Layer (HTML/CSS/JS)
│   ├── Admin Dashboard
│   ├── Analyst Tools
│   ├── Viewer Portal
│   └── Chart Creator
├── API Layer (FastAPI)
│   ├── Authentication & Authorization
│   ├── Dashboard Routes
│   ├── Data Routes
│   └── Export Routes
├── Business Logic Layer
│   ├── Dashboard Generator
│   ├── Chart Generator
│   ├── NLU Pipeline
│   └── RAG System
└── Data Layer
    ├── MySQL Database
    ├── File Storage (exports)
    └── Model Storage (ML models)
```

## Project Structure

```
Gen-Dash/
├── Frontend/                      # Web interface files
│   ├── admin_dashboard.html       # Admin interface
│   ├── analyst_dashboard.html     # Analyst interface
│   ├── chart_creator.html         # Dashboard chart builder
│   ├── viewer_dashboard.html      # Public dashboard viewer
│   ├── style.css                  # Global stylesheet
│   └── [other HTML pages]
├── auth/                          # Authentication module
│   ├── auth.py                    # Core auth logic
│   ├── add_department_column.py   # Department management
│   └── requirements_auth.txt
├── dashboard/                     # Dashboard generation
│   ├── dashboard_generator.py     # Main dashboard engine
│   ├── interactive_dashboard.py   # Interactive features
│   ├── dashboard_explainer.py     # AI-powered explanations
│   ├── add_dashboard_table.py     # Table integration
│   └── [related modules]
├── charts/                        # Chart generation
│   ├── chart_generator.py         # Chart creation engine
│   └── data_connector.py          # Data source connection
├── database/                      # Data models
│   ├── models.py                  # SQLAlchemy models (User, Dashboard)
│   └── schemas.py                 # Pydantic validation schemas
├── nlu/                           # Natural Language Understanding
│   ├── predict_ner.py             # NER prediction
│   ├── train_ner.py               # NER model training
│   ├── train.py                   # Intent classifier training
│   ├── test_ner.py                # NER tests
│   └── chart_pipeline.py          # Custom NLU pipeline
├── rag/                           # Retrieval-Augmented Generation
│   ├── rag.py                     # RAG system engine
│   ├── loaders.py                 # Multi-format document loader
│   └── requirements_rag.txt
├── ner_model/                     # Pre-trained NER model
│   ├── model.safetensors          # Model weights
│   ├── config.json                # Model configuration
│   └── [tokenizer files]
├── data/                          # Training and sample data
│   ├── training_data.json         # Labeled training data
│   ├── annotate_data.json         # Annotation data
│   └── advanced_ecommerce_training.json
├── exports/                       # Exported data files
│   └── [dated CSV exports]
├── saved_dashboards/              # Persisted dashboard HTML
│   └── [dashboard snapshots]
├── temp_dashboards/               # Temporary working dashboards
│   └── [session dashboards]
├── main.py                        # FastAPI application entry point
├── requirements.txt               # Python dependencies
├── .env                           # Environment configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- MySQL Server 5.7 or higher
- pip (Python package manager)
- Git (for version control)
- 2GB RAM minimum
- 500MB disk space

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Gen-Dash.git
cd Gen-Dash
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. MySQL Database Setup

```bash
# Create database and user
mysql -u root -p
```

```sql
CREATE DATABASE analytics_dashboard CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'appuser'@'localhost' IDENTIFIED BY 'YourStrongPassword';
GRANT ALL PRIVILEGES ON analytics_dashboard.* TO 'appuser'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

#### 5. Configure Environment Variables

Create/update `.env` file:

```env
# Database Configuration
DATABASE_URL=mysql+pymysql://appuser:YourStrongPassword@localhost/analytics_dashboard

# JWT Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# OpenAI/LangChain Configuration (for RAG)
OPENAI_API_KEY=your-openai-key
```

#### 6. Initialize Database

```bash
python -c "from database.models import Base, engine; Base.metadata.create_all(bind=engine)"
```

#### 7. Run the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the application at: `http://localhost:8000`

## Configuration

### Database Configuration

Update the `DATABASE_URL` in `.env`:

```python
# SQLite (Development)
DATABASE_URL=sqlite:///./analytics_dashboard.db

# MySQL (Production)
DATABASE_URL=mysql+pymysql://user:password@localhost/analytics_dashboard

# PostgreSQL (Enterprise)
DATABASE_URL=postgresql://user:password@localhost/analytics_dashboard
```

### Theme Configuration

Supported themes in `dashboard_generator.py`:

- `modern`: Blue and purple gradient (default)
- `dark`: Dark mode with bright accents
- `light`: Light professional theme
- `corporate`: Enterprise blue theme

### NLP Model Configuration

The NER model is loaded from `ner_model/` directory. Retrain using:

```bash
python nlu/train_ner.py --data data/training_data.json --epochs 10
```

## Usage

### Basic Workflow

#### 1. User Registration & Login

Navigate to `/register.html` to create an account. Upon registration, users are assigned the "Viewer" role by default.

#### 2. Dashboard Creation

As an **Analyst** or **Admin**:

1. Go to Chart Creator (`/chart_creator.html`)
2. Select data source and chart type
3. Configure visualization options
4. Preview and save dashboard
5. Share with specific users/departments

#### 3. Dashboard Viewing

As a **Viewer**:

1. Access your assigned dashboards
2. Interact with charts (zoom, filter, hover)
3. Export data or full dashboard
4. View AI-powered insights

### API Endpoints

#### Authentication

```
POST   /api/auth/register          - Register new user
POST   /api/auth/login             - Login and create session
POST   /api/auth/logout            - Logout and clear session
GET    /api/auth/profile           - Get current user profile
```

#### Dashboards

```
GET    /api/dashboards             - List user's dashboards
POST   /api/dashboards             - Create new dashboard
GET    /api/dashboards/{id}        - Get dashboard details
PUT    /api/dashboards/{id}        - Update dashboard
DELETE /api/dashboards/{id}        - Delete dashboard
GET    /api/dashboards/{id}/export - Export dashboard
```

#### Charts

```
POST   /api/charts/generate        - Generate chart from data
GET    /api/charts/{id}            - Get chart details
```

#### Data

```
POST   /api/data/upload            - Upload data file
GET    /api/data/{id}/preview      - Preview data
GET    /api/data/{id}/export       - Export data as CSV
```

#### NLU/AI Features

```
POST   /api/nlu/predict            - Predict entities in text
POST   /api/rag/query              - Query with RAG system
```

## Role-Based Access Control

| Role                | Permissions                                         | Scope               |
| ------------------- | --------------------------------------------------- | ------------------- |
| **Admin**           | Full system access, user management, all dashboards | Organization-wide   |
| **Analyst**         | Create/edit dashboards, manage charts, upload data  | Department-wide     |
| **Department Head** | View department dashboards, manage team             | Department-specific |
| **Viewer**          | View assigned dashboards only, export data          | Personal/shared     |

### Permission Matrix

```
Feature                 Admin    Analyst    Dept Head    Viewer
─────────────────────────────────────────────────────────────
Create Dashboard        Yes      Yes        No           No
Edit Dashboard          Yes      Yes*       No           No
Delete Dashboard        Yes      Yes*       No           No
Share Dashboard         Yes      Yes        Yes*         No
View Dashboard          Yes      Yes        Yes          Yes*
Export Data             Yes      Yes        Yes          Yes*
Manage Users            Yes      No         No           No
Manage Departments      Yes      No         No           No

* Limited to own/assigned resources
```

## Advanced Features

### Natural Language Dashboard Creation

Create dashboards using natural language:

```python
# Example: User inputs "Show me total sales by region for Q4"
# NLU system parses intent and entities
# Dashboard is auto-generated with appropriate visualizations
```

### AI-Powered Dashboard Explanations

Get AI-generated insights about your data:

```
GET /api/dashboards/{id}/explain

Response:
{
  "summary": "Sales increased 23% in Q4",
  "trends": ["Strong growth in North America", "Slight decline in APAC"],
  "anomalies": ["Unexpected spike in Europe on Dec 15"],
  "recommendations": ["Investigate Europe spike", "Scale North America resources"]
}
```

### Multi-Document RAG System

Query across multiple documents:

```python
POST /api/rag/query
{
  "question": "What are the company policies on remote work?",
  "documents": ["policy_handbook.pdf", "hr_guide.docx"]
}
```

### Custom Chart Types

Supported visualizations:

- Area Charts
- Bar Charts
- Line Charts
- Pie/Donut Charts
- Radar/Polar Charts
- Scatter Plots
- Heatmaps
- Sunburst Diagrams
- Waterfall Charts

## Data Management

### Supported Data Formats

- **Input**: CSV, Excel, JSON, SQL queries
- **Output**: CSV, Excel, HTML, JSON, PNG (chart export)

### Data Export

```bash
# Export dashboard as interactive HTML
GET /api/dashboards/{id}/export?format=html

# Export underlying data as CSV
GET /api/dashboards/{id}/data/export?format=csv

# Export with timestamp
GET /api/dashboards/{id}/export?include_timestamp=true
```

### Data Security

- All exports are timestamped and logged
- Access controlled by role-based permissions
- Data stored securely with password hashing
- CORS enabled for specific origins only

## Development

### Project Setup for Developers

```bash
# Clone repository
git clone <repo-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Configure Git hooks
git config core.hooksPath .githooks
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v

# Run NER tests
python nlu/test_ner.py
```

### Code Quality

```bash
# Format code with Black
black .

# Lint with Flake8
flake8 . --max-line-length=100

# Type checking with Mypy
mypy . --ignore-missing-imports
```

### Building Documentation

```bash
# Generate API documentation
python -m mkdocs build

# View docs locally
python -m mkdocs serve
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions/classes
- Add type hints where applicable
- Keep functions small and focused
- Write unit tests for new code
- Maintain test coverage above 80%

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

```
Error: Cannot connect to MySQL server
```

**Solution:**

```bash
# Check MySQL is running
mysql -u root -p

# Verify connection string in .env
# Ensure database and user exist
```

#### 2. Port Already in Use

```
Error: Address already in use
```

**Solution:**

```bash
# Use different port
uvicorn main:app --port 8001

# Or kill process on port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
```

#### 3. NER Model Not Found

```
Error: ner_model directory not found
```

**Solution:**

```bash
# Ensure ner_model directory exists with pre-trained weights
# Or retrain the model
python nlu/train_ner.py --data data/training_data.json
```

#### 4. CORS Errors

**Solution:**

- Update CORS settings in `main.py` line 70-77
- Add frontend URL to `allow_origins` list

#### 5. Memory Issues

```
Error: Out of memory during chart generation
```

**Solution:**

```bash
# Increase Python memory limit
export PYTHONUNBUFFERED=1

# Or use smaller batch sizes in chart generation
```

### Debug Mode

Enable debug logging:

```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Caching

The system implements caching for:

- Dashboard metadata (5 minutes)
- Chart data (10 minutes)
- User permissions (session duration)

### Database Indexing

Recommended indexes:

```sql
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_dashboard_owner ON dashboards(owner_id);
CREATE INDEX idx_dashboard_department ON dashboards(department_id);
CREATE INDEX idx_dashboard_created ON dashboards(created_at);
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/api/dashboards

# Using locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support & Contact

- **Documentation**: [Read the Docs](https://gen-dash.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/Gen-Dash/issues)
- **Email**: support@gen-dash.io
- **Slack**: [Join Community](https://join.slack.com/t/gen-dash/shared_invite/xyz)

## Acknowledgments

- **FastAPI**: Modern web framework
- **Plotly**: Interactive visualization library
- **Transformers**: NLP model library
- **LangChain**: RAG system framework
- **SQLAlchemy**: ORM framework
- **MySQL**: Database engine

## Roadmap

### Q1 2026

- [ ] Real-time collaborative dashboards
- [ ] Advanced predictive analytics
- [ ] Mobile app (React Native)

### Q2 2026

- [ ] Multi-language support
- [ ] Advanced scheduling/automation
- [ ] Dashboard versioning system

### Q3 2026

- [ ] Cloud deployment templates (AWS/GCP/Azure)
- [ ] Advanced permission models
- [ ] Custom chart development SDK

---

**Version**: 1.0.0  
**Last Updated**: December 19, 2025  
**Maintained By**: Gen-Dash Team

For more information, visit [gen-dash.io](https://gen-dash.io)
