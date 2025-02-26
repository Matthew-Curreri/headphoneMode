#!/bin/bash

DB_PATH="../backend/headphone-mode.db"
SCHEMA_FILE="../Setup-Scripts/schema.sql"

# Ensure SQLite is installed
if ! command -v sqlite3 &> /dev/null; then
    echo "SQLite3 is not installed. Please install it and run this script again."
    exit 1
fi

# Check if the database exists
if [ -f "$DB_PATH" ]; then
    echo "Database found at $DB_PATH. Checking structure..."
    
    # Check if required tables exist
    TABLE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('users', 'sessions', 'preferences', 'security', 'usage_tracking', 'subscriptions');")

    if [ "$TABLE_COUNT" -eq 12 ]; then
        echo "All required tables are present. No changes needed."
        exit 0
    else
        echo "Some tables are missing. Applying schema..."
        sqlite3 "$DB_PATH" < "$SCHEMA_FILE"
    fi
else
    echo "Database not found. Creating new database and applying schema..."
    sqlite3 "$DB_PATH" < "$SCHEMA_FILE"
fi

echo "Database setup complete."
