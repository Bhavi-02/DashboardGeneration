# Changelog

All notable changes to Gen-Dash will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-19

### 🐛 Fixed

- Fixed 20+ bare `except:` clauses with specific exception handling
  - `charts/data_connector.py`: Fixed year filter, numeric column detection, alternate header detection
  - `dashboard/dashboard_generator.py`: Fixed numpy array conversion, sample chart creation
  - `charts/chart_generator.py`: Fixed heatmap fallback handling
  - `rag/rag.py`: Fixed temp file cleanup exception handling
- Fixed potential database connection leaks in all API endpoints
- Fixed missing type hints for SQLAlchemy Session

### ✨ Added

- **Database Connection Pooling**
  - Pool size: 5 connections
  - Max overflow: 10 connections
  - Connection recycling: Every hour
  - Health checks: Enabled (pool_pre_ping)
- **Structured Logging Framework**

  - New module: `utils/logger.py`
  - Module-specific loggers (app, database, auth, charts, dashboard, nlu, rag)
  - Rotating file handlers (10MB max, 5 backups)
  - Console and file output
  - Log directory: `logs/`

- **Input Validation Framework**

  - New module: `utils/validators.py`
  - Email validation with regex
  - Username validation (3-50 chars, alphanumeric + \_-)
  - Password validation (8-72 chars for bcrypt)
  - Role validation
  - Filename sanitization (prevents directory traversal)
  - Integer validation with bounds
  - HTML escaping (XSS prevention)

- **Documentation**

  - Added `OPTIMIZATIONS.md` - Comprehensive optimization guide
  - Added `docs/QUICK_REFERENCE.md` - Quick reference for logging and validation
  - Added `logs/README.md` - Logs directory documentation
  - Added `CHANGELOG.md` - This file

- **Database Dependency Injection**
  - New `get_db()` dependency for automatic session management
  - Automatic connection cleanup in all endpoints

### 🔧 Changed

- Updated `main.py` imports to include logging and Session type
- Enhanced error messages with better context and debugging info
- Improved exception handling with specific exception types
- Updated README.md with v1.1.0 announcement
- Changed build status from "Active Development" to "Production Ready"

### 🔒 Security

- Added input sanitization to prevent XSS attacks
- Added filename sanitization to prevent directory traversal
- Implemented HTML escaping for user inputs
- Better error handling prevents information leakage

### ⚡ Performance

- Database connection pooling reduces connection overhead
- Connection recycling prevents stale connections
- Health checks reduce failed query attempts
- Automatic connection cleanup prevents resource leaks

### 🧹 Maintenance

- Better error tracking with structured logging
- Improved code quality with specific exception handling
- Type hints added for better IDE support
- Centralized validation logic

### 📝 Internal

- Created `utils/` directory for shared utilities
- Created `logs/` directory for application logs
- Added logs to `.gitignore` (already present)

---

## [1.0.0] - 2025-12-01

### Initial Release

#### Features

- FastAPI-based backend
- Role-based access control (Admin, Analyst, Departmental, Viewer)
- Natural Language Understanding with DistilBERT NER
- Interactive dashboard generation with Plotly
- Multi-theme support (Modern, Dark, Light, Corporate)
- RAG system with FAISS and LangChain
- MySQL database with SQLAlchemy ORM
- Multi-dataset support
- Chart export (HTML, CSV)
- Dashboard snapshots
- User authentication with bcrypt

#### Components

- Authentication system (`auth/`)
- Chart generation (`charts/`)
- Dashboard creation (`dashboard/`)
- NLU pipeline (`nlu/`)
- RAG system (`rag/`)
- Database models (`database/`)
- Frontend interfaces (`Frontend/`)

---

## Version Naming

- **Major version (X.0.0)**: Breaking changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

---

## Links

- [Repository](https://github.com/yourusername/Gen-Dash)
- [Documentation](README.md)
- [Optimizations Guide](OPTIMIZATIONS.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)
