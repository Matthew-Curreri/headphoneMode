#!/bin/bash
# HeadphoneMode Project Setup Script
# This script creates the necessary directory structure and files for the HeadphoneMode project.
# It uses variables for directories and files to allow easy expansion and modification.

# Define base directories
frontend_dir="frontend"
backend_dir="backend"
other_dirs=("localAudioProc" "whisper" "temp" "Setup-Scripts")

# Define subdirectories for each base directory
frontend_subdirs=("media/icons" "media/images" "scripts" "styles")
backend_subdirs=("pages/api" "utils")

# Define files to create in each directory
frontend_files=("index.html" "styles/styles.css" "scripts/main.js" "scripts/init.js" "scripts/api.js" "scripts/audio.js" "scripts/clientsideGPU.js")
backend_files=("config.js" "headphoneMode.db" "pages/index.js" "pages/api/audio.js" "pages/api/openai.js" \
               "pages/api/preferences.js" "pages/api/session.js" "utils/prompts.js" "utils/sessionHandler.js" \
               "utils/webServer.js")
local_audio_proc_files=("keywordScan.js" "micAccess.js")
setup_scripts_files=("buildDir.sh" "database.sh" "init.sh" "schema.sql" "setUpDB.sh")
whisper_files=("whisperSetup.sh")
root_files=(".env" "README.md" "DATABASE.md" "package.json")

# Create base directories
mkdir -p "$frontend_dir" "$backend_dir"
for dir in "${other_dirs[@]}"; do
    mkdir -p "$dir"
done

# Create subdirectories for frontend and backend
for subdir in "${frontend_subdirs[@]}"; do
    mkdir -p "$frontend_dir/$subdir"
done

for subdir in "${backend_subdirs[@]}"; do
    mkdir -p "$backend_dir/$subdir"
done

# Create necessary files in the frontend directory
for file in "${frontend_files[@]}"; do
    touch "$frontend_dir/$file"
done

# Create necessary files in the backend directory
for file in "${backend_files[@]}"; do
    touch "$backend_dir/$file"
done

# Create necessary files in localAudioProc directory
for file in "${local_audio_proc_files[@]}"; do
    touch "localAudioProc/$file"
done

# Create necessary files in Setup-Scripts directory
for file in "${setup_scripts_files[@]}"; do
    touch "Setup-Scripts/$file"
done

# Create necessary files in whisper directory
for file in "${whisper_files[@]}"; do
    touch "whisper/$file"
done

# Create necessary files at the project root
for file in "${root_files[@]}"; do
    touch "$file"
done

echo "Project structure setup complete."
