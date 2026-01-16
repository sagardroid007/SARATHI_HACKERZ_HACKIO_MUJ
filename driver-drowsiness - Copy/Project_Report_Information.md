# SARATHI AI: Smart Driving Companion - Project Report Data

This document contains comprehensive information about the **Sarathi AI** project, organized to assist in the preparation of a final project report.

---

## 1. Project Overview
**Sarathi AI** is an intelligent, real-time safety and assistance system designed for drivers. It combines **Computer Vision** (for drowsiness detection) with a **Voice Assistant** (for hands-free control and information) and a **Premium Dashboard** (for visual monitoring).

### Key Objectives:
- **Safety**: Monitor driver fatigue and provide instant alerts.
- **Convenience**: Provide a voice-controlled interface ("Sara") for weather, navigation, and general queries.
- **Integration**: Seamlessly connect with LLMs (Gemini), Weather APIs, and Navigation services.

---

## 2. Technical Stack

### Core Technologies:
- **Programming Language**: Python 3.10 - 3.12
- **Computer Vision**: OpenCV, dlib (68-point facial landmark detection)
- **Speech & AI**:
    - **Wake Word Detection**: Picovoice Porcupine ("Hey Sara", "Bumblebee")
    - **Speech-to-Text**: Google Speech Recognition API
    - **Text-to-Speech**: pyttsx3
    - **Natural Language**: Gemini 1.5 Flash (LLM), spaCy (Intent extraction)
- **User Interface**: CustomTkinter (Modern, dark-themed GUI)
- **External Services**: OpenWeatherMap API, Google Maps, Telegram API (for alerts)

---

## 3. System Architecture & Modules

### 3.1. Drowsiness Detection (`drowsiness_detect.py` / `main.py`)
- **Mechanism**: Calculates the **Eye Aspect Ratio (EAR)** using 6 facial landmarks per eye.
- **Thresholds**:
    - `EAR_THRESHOLD`: **0.20** (Values below this indicate closed eyes).
    - `CONSEC_FRAMES`: **30** (Approximately 1 second of closed eyes triggers the alert).
- **Actions**:
    - Plays a high-pitched alert audio (`alert.wav`).
    - Sends a real-time notification to a **Telegram Bot** to alert remote monitors.
    - Displays a "DROWSY!" warning on the dashboard.

### 3.2. Voice Assistant ("Sara")
- **Wake Word**: Users activate the assistant by saying **"Hey Sara"**.
- **Interaction Flow**:
    1. **Wake Word Listener**: Independent thread monitoring audio for the keyword.
    2. **STT**: Captures user voice and converts it to text using Google's API.
    3. **Processing**: 
        - Local intent matching via `cms.json` for fixed commands.
        - LLM fallback to **Gemini 1.5 Flash** for complex queries.
    4. **TTS**: Responses are spoken back to the driver clearly.

### 3.3. GUI Dashboard (`gui_dashboard.py`)
- **Theme**: "Deep Void Black" with "Neon Green" and "Cyber Cyan" accents for a premium feel.
- **Components**:
    - **Live Video Feed**: With safety overlays (EAR, FPS, Status).
    - **System Status Sidebar**: Real-time indicators for Voice, AI Mode, and System health.
    - **Info Cards**: Dynamic widgets for Weather and Navigation status.

### 3.4. Command Knowledge Base (`cms.json`)
The system contains a pre-defined knowledge base of **440+ commands** including:
- **General Queries**: "What are you?", "Who is Sara?"
- **Entertainment**: Jokes, puns, and music requests.
- **Driving Education**: Fact about traffic patterns, fuel efficiency, and iconic cars.
- **Utility**: Navigation and Weather triggers.

### 3.5. Utility Clients
- **`weather_client.py`**: Fetches real-time weather using OpenWeatherMap.
- **`maps_client.py`**: Automatically opens Google Maps navigation in the default browser upon voice command.
- **`gemini_client.py`**: Handles integration with Google Generative AI for smart conversations.

---

## 4. Key Logic Details (For Implementation Section)

### EAR Calculation Formula:
The distance between the vertical eye landmarks is divided by the distance between horizontal landmarks.
```python
ear = (dist(P2, P6) + dist(P3, P5)) / (2 * dist(P1, P4))
```

### Threading Architecture:
To ensure the GUI remains responsive while processing video and audio:
- **Main Thread**: Runs the CustomTkinter GUI loop.
- **Core Loop Thread**: Handles OpenCV video capture and EAR processing.
- **Wake Word Thread**: Constantly listens for "Hey Sara" without blocking video.
- **Assistant Worker**: Handles STT, LLM queries, and TTS in the background.

---

## 5. Setup & Future Scope

### Setup Requirements:
- Python dependencies (OpenCV, dlib, Picovoice, Pygame).
- API Keys for Gemini, OpenWeather, and Picovoice.
- Dlib shape predictor model (`shape_predictor_68_face_landmarks.dat`).

### Future Scope:
- **Head Pose Estimation**: Detecting if the driver is looking away from the road.
- **Yawn Detection**: Using mouth aspect ratio (MAR) as an early sign of fatigue.
- **Emotion Recognition**: To detect driver stress levels.
- **Offline LLM**: For better privacy and zero-latency responses.
