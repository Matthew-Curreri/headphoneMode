#!/bin/bash

# Define the base directory
BASE_DIR="headphone-mode"

# Define the directory structure
DIRECTORIES=(
    "$BASE_DIR/frontend/media/images"
    "$BASE_DIR/frontend/media/icons"
    "$BASE_DIR/frontend/scripts"
    "$BASE_DIR/frontend/styles"
    "$BASE_DIR/backend/pages/api"
    "$BASE_DIR/backend/utils"
)

# Define the files to create
FILES=(
    "$BASE_DIR/frontend/index.html"
    "$BASE_DIR/frontend/styles/styles.css"
    "$BASE_DIR/frontend/scripts/main.js"
    "$BASE_DIR/frontend/scripts/init.js"
    "$BASE_DIR/frontend/scripts/api.js"
    "$BASE_DIR/frontend/scripts/audio.js"
    "$BASE_DIR/backend/pages/index.js"
    "$BASE_DIR/backend/pages/api/session.js"
    "$BASE_DIR/backend/pages/api/preferences.js"
    "$BASE_DIR/backend/pages/api/openai.js"
    "$BASE_DIR/backend/utils/prompts.js"
    "$BASE_DIR/backend/utils/sessionHandler.js"
    "$BASE_DIR/backend/pages/api/audio.js"
    "$BASE_DIR/backend/config.js"
    "$BASE_DIR/package.json"
    "$BASE_DIR/.env"
    "$BASE_DIR/README.md"
)

# Create directories
for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
done

# Create files
for file in "${FILES[@]}"; do
    touch "$file"
done

echo "Project structure created successfully."

