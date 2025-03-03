# Headphone Mode

Headphone Mode is an LLM-powered experience designed to enhance user interaction by providing contextual isolation, adaptive filtering, and persistent session management. The project is built with a Vanilla JavaScript frontend and a Node.js backend (using Express), utilizing the OpenAI API for processing. SQLite is planned for session persistence and user preference storage.
website: www.headphoneMode.com
blog: www.curreri.world
## Features

- **Contextual Isolation**: Focused, uninterrupted interaction with the LLM.
- **Adaptive Filtering**: Custom response styles for different use cases.
- **State Persistence**: Saves and recalls previous interactions.
- **API-Optimized Backend**: Efficient session and preference handling.
- **Minimalist UI**: Designed for a distraction-free experience.
- **Audio Processing (Planned)**: Voice input with OpenAI Whisper transcription.
- **Client side LLM processing**: Off load training and generation to clientside.  **Keep up with the Jones w/o billions of dollars**

## Tech Stack

- **Frontend**: Vanilla JavaScript (HTML, CSS, JS)
- **Backend**: Node.js with Express (API routes for session and preference handling)
- **Database**: SQLite (for persistent session storage)
- **LLM**: OpenAI API
- **Audio Processing**: OpenAI Whisper (planned integration)

## Installation

```sh
# Clone the repository
git clone https://github.com/Matthew-Curreri/headphoneMode
cd headphoneMode

# Install dependencies
npm install

# Set environment variables
cp .env.example .env

# Start the development server to do
npm run dev
```

## Backend Overview

- ``: Main entry point, sets up the Express server, API routes, and integrates OpenAI API.
- ``: Handles AI chat interactions.
- `` (planned): Processes user audio via Whisper API.
- ``: Manages SQLite database interactions (sessions, preferences, etc.).
- ``: Stores configuration values (e.g., API keys, database path).

## Planned Features

- **Voice Input**: Capture microphone input via WebRTC and send it for transcription.
- **Audio Output**: Text-to-speech playback for AI responses.
- **Enhanced Session Management**: More robust session tracking for persistent interactions.
- **Adaptive Filtering**: Dynamic response filtering based on user preferences.

## Authors

- **mcurreri** (Project Lead)
- **ChatGPT** (Co-Author)

## License

MIT License

## To-Do

- Implement frontend UI with preset toggles.
- Complete session handling logic.
- Develop iPhone and Android frontends.
- Optimize OpenAI API usage for cost efficiency.
- Implement local audio processing for speech-to-text.

For any inquiries or contributions, feel free to open an issue or submit a pull request!

