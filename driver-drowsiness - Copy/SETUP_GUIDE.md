# Sarathi AI - Setup & Deployment Guide

Follow these steps to run **Sarathi AI** on any Windows device.

## 1. Prerequisites
*   **Python 3.10 - 3.12**: [Download here](https://www.python.org/downloads/)
*   **API Keys**:
    *   **Gemini**: Get a free key at [AI Studio](https://aistudio.google.com/).
    *   **OpenWeather**: Get a free key at [OpenWeatherMap](https://openweathermap.org/api).
    *   **Porcupine**: Get an AccessKey at [Picovoice Console](https://console.picovoice.ai/).

## 2. Installation
1.  **Clone/Copy** the project folder to the new device.
2.  Open **Command Prompt** (cmd) or **PowerShell** inside the folder.
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

> [!IMPORTANT]
> **Dlib Installation Note**: 
> If `pip install dlib` fails, you may need to install **CMake** and **Visual Studio C++ Build Tools**, or download a pre-compiled `.whl` file for your Python version.

## 3. Configuration
1.  Open `gemini_client.py` and paste your **Gemini API Key**.
2.  Open `weather_client.py` and paste your **Weather API Key**.
3.  Open `main.py` and update the `PICOVOICE_ACCESS_KEY` if needed.

## 4. Running the App
Simply double-click:
**`start_assistant.bat`**

This script will automatically detect if a virtual environment exists. If not, it will default to your system Python.

## 5. Controls
*   **'q'**: Quit
*   **'o'**: Toggle Safety Overlay
*   **Voice**: Say "Hey Sara" or "Hey Sarathi" to activate the assistant.
