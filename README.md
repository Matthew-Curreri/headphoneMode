Headphone Mode is an LLM-powered experience designed to enhance user interaction by providing contextual isolation, adaptive filtering, and persistent session management. The project is built with a Vanilla JavaScript frontend and a Next.js backend, utilizing the OpenAI API for processing.

Features

Contextual Isolation: Focused, uninterrupted interaction with the LLM.

Adaptive Filtering: Custom response styles for different use cases.

State Persistence: Saves and recalls previous interactions.

API-Optimized Backend: Efficient session and preference handling.

Minimalist UI: Designed for a distraction-free experience.

Tech Stack

Frontend: Vanilla JavaScript (HTML, CSS, JS)

Backend: Next.js (API routes for session and preference handling)

Database: TBD (for persistent session storage if required)

LLM: OpenAI API

Installation

# Clone the repository
git clone https://github.com/Matthew-Curreri/headphoneMode
cd headphone-mode

# Install dependencies
npm installhttps://github.com/Matthew-Curreri/headphoneMode/

# Set environment variables
cp .env.example .env

# Start the development server
npm run dev

Folder Structure

/headphone-mode
├── /frontend
│   ├── /media
│   ├── /scripts
│   ├── /styles
│   ├── index.html
│
├── /backend
│   ├── pages/
│   │   ├── api/
│   ├── utils/
│   ├── config.js
│
├── package.json
├── .env
├── README.md

Authors

mcurreri (Project Lead)

ChatGPT (Co-Author)

License

MIT License

To-Do

Complete session handling logic.

Implement frontend UI with preset toggles.

Develop iPhone and Android frontends.

Optimize OpenAI API usage for cost efficiency.


For any inquiries or contributions, feel free to open an issue or submit a pull request!
