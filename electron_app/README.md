# TheNotes - Desktop Speech-to-Text Client

A desktop application for real-time speech-to-text transcription built with Electron.

## Overview

TheNotes is a client application that connects to a separate Python server for speech-to-text processing. This package contains only the client-side code that provides the user interface and handles audio recording.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Client](#starting-the-client)
  - [Building the Application](#building-the-application)


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

## Prerequisites

- Node.js (v16 or higher) - this will automatically install Electron
- npm or yarn
- Python server running on `localhost:8000` (separate repository)

## Installation

```bash
cd electron_app
npm install
```

## Usage

### Starting the Client

1. First, make sure your Python [server](../examples/server.py) is running on `localhost:8000`
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

---

**Note**: This is the client application only. You need to run the Python server separately for full functionality.