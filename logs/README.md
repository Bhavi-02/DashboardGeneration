# Logs Directory

This directory contains application log files.

## Log Files

- `gendash.log` - Main application logs
- `gendash_database.log` - Database operation logs
- `gendash_auth.log` - Authentication logs
- `gendash_charts.log` - Chart generation logs
- `gendash_dashboard.log` - Dashboard generation logs
- `gendash_nlu.log` - NLU pipeline logs
- `gendash_rag.log` - RAG system logs

## Log Rotation

Logs are automatically rotated when they reach 10MB. Up to 5 backup files are kept.

## .gitignore

Log files are excluded from git tracking to prevent sensitive information leakage.
