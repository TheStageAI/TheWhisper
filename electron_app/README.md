# TheNotes - Desktop Speech-to-Text Client

A modern desktop application for real-time speech-to-text transcription built with Electron.

## Overview

TheNotes is a client application that connects to a separate Python server for speech-to-text processing. This repository contains only the client-side code that provides the user interface and handles audio recording.

## Prerequisites

- Node.js (v16 or higher) - this will automatically install Electron
- npm or yarn
- Python server running on `localhost:8000` (separate repository)

**Note**: You don't need to install Electron separately - it will be installed automatically with the dependencies.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd electron_app
```

2. Install dependencies:
```bash
npm install
```

This will install all required dependencies including Electron, SiriWave animation library, and the necessary audio processing libraries. Electron will automatically download and install Chromium for you.

## Usage

### Starting the Client

1. First, make sure your Python server is running on `localhost:8000`
2. Start the client application:
```bash
npm start
```

For development mode:
```bash
npm run start:dev
```

### Building the Application

To build the application locally:

```bash
npm run build
```

This will create a local build in the `dist/` directory. Note that the built application will not be signed and may show security warnings when running.

## Development

### Project Structure

```
electron_app/
├── main.js           # Electron main process
├── app.js            # Frontend application logic
├── preload.js        # Secure IPC bridge
├── audio-processor.js # Audio processing
├── index.html        # Main HTML template
├── styles.css        # Application styles
└── assets/           # Static assets (fonts, icons)
    ├── fonts/
    └── icon.png
```

## Dependencies

- **Electron**: Desktop application framework
- **electron-log**: Logging functionality
- **SiriWave**: Animation library (installed via npm from [kopiro/siriwave](https://github.com/kopiro/siriwave))

## Security

The application implements several security measures:

- Context isolation enabled
- Node integration disabled in renderer
- Sandbox mode enabled
- Content Security Policy configured
- Secure IPC communication via preload script

**Note**: The built application is not code-signed and may show security warnings on macOS. This is normal for development builds.

---

**Note**: This is the client application only. You need to run the Python server separately for full functionality.