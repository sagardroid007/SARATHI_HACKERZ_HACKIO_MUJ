import time
import cv2
import dlib
import pygame
from imutils import face_utils
from scipy.spatial import distance
import json
import re
import difflib
import pyttsx3
import pygame
import pyaudio
import spacy
import speech_recognition as sr
import threading
import numpy as np
import os
from datetime import datetime
import struct
import pvporcupine

# ensure pygame mixer is initialized and alert audio is loaded
ALERT_AVAILABLE = False
try:
    pygame.mixer.init()
    ALERT_PATH = os.path.join(os.path.dirname(__file__), "audio", "alert.wav")
    if os.path.isfile(ALERT_PATH):
        try:
            pygame.mixer.music.load(ALERT_PATH)
            ALERT_AVAILABLE = True
        except Exception as e:
            print(f"[WARN] Failed to load alert audio '{ALERT_PATH}':", e)
    else:
        print(f"[WARN] Alert audio not found at: {ALERT_PATH}")
except Exception as e:
    print("[WARN] pygame.mixer.init() failed:", e)


# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")


# Load cms.json commands (use question field as primary match text)
with open(r'cms.json', 'r', encoding='utf-8') as jf:
    cms_data = json.load(jf)
commands = cms_data.get("commands", [])

# deterministic normalization (expand common SMS abbreviations)
_ABBR_MAP = {
    "u": "you", "r": "are", "ur": "your", "ya": "you",
    "whats": "what is", "what's": "what is", "whatsup": "what is up",
    "wanna": "want to", "gonna": "going to", "cant": "cannot",
    "im": "i am", "id": "i would"
}

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [ _ABBR_MAP.get(t, t) for t in s.split() ]
    return " ".join(tokens)

def extract_intents(user_input):
    # simple token set (kept for compatibility)
    return set(normalize_text(user_input).split())

def best_command_match(user_input, commands_list, overlap_threshold=0.5, fuzzy_threshold=0.65):
    """
    Match user_input to the best command from cms.json.
    Priority: exact normalized question -> substring -> token-overlap -> fuzzy ratio.
    Returns (command_dict or None, score)
    """
    user_norm = normalize_text(user_input)
    candidates = []
    for cmd in commands_list:
        cand = normalize_text(cmd.get("question") or cmd.get("response") or "")
        candidates.append((cmd, cand))

    # exact normalized match
    for cmd, cand in candidates:
        if cand and cand == user_norm:
            return cmd, 1.0

    # substring
    for cmd, cand in candidates:
        if cand and (user_norm in cand or cand in user_norm):
            return cmd, 0.95

    # token overlap
    user_tokens = set(user_norm.split())
    best_cmd = None
    best_score = 0.0
    for cmd, cand in candidates:
        if not cand:
            continue
        cmd_tokens = set(cand.split())
        if not cmd_tokens:
            continue
        overlap = len(user_tokens & cmd_tokens) / len(cmd_tokens)
        if overlap > best_score:
            best_score = overlap
            best_cmd = cmd
    if best_cmd and best_score >= overlap_threshold:
        return best_cmd, best_score

    # fuzzy fallback
    best_ratio = 0.0
    best_fuzzy = None
    for cmd, cand in candidates:
        if not cand:
            continue
        ratio = difflib.SequenceMatcher(None, user_norm, cand).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_fuzzy = cmd
    if best_fuzzy and best_ratio >= fuzzy_threshold:
        return best_fuzzy, best_ratio

    return None, max(best_score, best_ratio, 0.0)

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.2

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

# Counts no. of consecutive frames below threshold value
COUNTER = 0

# Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Load the JSON data
with open('cms.json', 'r') as json_file:
    commands = json.load(json_file)["commands"]

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")


# This function calculates and returns eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear


# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Start webcam video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# UI / overlay settings
SHOW_OVERLAY = True
FULLSCREEN = False
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

_prev_time = time.time()
_fps = 0.0

def draw_overlay(frame, ear, drowsy, fps, audio_level=0.0):
    """Draw semi-transparent status bar, EAR, FPS, instructions and live audio meter on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # status bar background (semi-transparent)
    bar_h = 90
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # EAR and drowsy status
    status_text = "DROWSY" if drowsy else "AWAKE"
    status_color = (0, 0, 255) if drowsy else (0, 200, 0)
    cv2.putText(frame, f"Status: {status_text}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # Audio meter (vertical bar on right side)
    meter_h = int(bar_h * 0.75)
    meter_w = 18
    meter_x = w - 30
    meter_y = 20
    # background
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
    # level
    level_h = int(audio_level * meter_h)
    level_y0 = meter_y + (meter_h - level_h)
    # color gradient: low=green -> mid=yellow -> high=red
    if audio_level < 0.6:
        col = (0, 200, 0)
    elif audio_level < 0.85:
        col = (0, 200, 200)
    else:
        col = (0, 0, 255)
    cv2.rectangle(frame, (meter_x, level_y0), (meter_x + meter_w, meter_y + meter_h), col, -1)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (200,200,200), 1)
    cv2.putText(frame, "Audio", (meter_x - 45, meter_y + meter_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # small help text
    help_text = "[w] Wake  [o] Toggle overlay  [s] Snapshot  [f] Fullscreen  [q] Quit"
    cv2.putText(frame, help_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    return frame

# Initialize Porcupine for wake word detection (robust)
wakeword_enabled = False
porcupine = None
audio_stream = None
pa = None
try:
    porcupine = pvporcupine.create(
        access_key="nQTUdktIxEO8BWG5sC45JKuafjf7fGtvbOSlEmybAYFPc9wIMhP+6g==",
        keyword_paths=['./Hey-Sara_en_windows_v3_0_0.ppn']  # ensure this file exists
    )
    print(f"[INFO] Porcupine OK (frame_length={porcupine.frame_length}, sample_rate={porcupine.sample_rate})")
    pa = pyaudio.PyAudio()
    try:
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        wakeword_enabled = True
    except Exception as e:
        print("[WARN] Failed to open input stream:", e)
        # list devices for debugging
        try:
            info = pa.get_host_api_info_by_index(0)
            dev_count = info.get('deviceCount', 0)
            print("[INFO] Audio devices:")
            for i in range(dev_count):
                dev = pa.get_device_info_by_host_api_device_index(0, i)
                print(f"  {i}: {dev.get('name')} (maxInputChannels={dev.get('maxInputChannels')})")
        except Exception as e2:
            print("[WARN] Could not list audio devices:", e2)
        audio_stream = None
except Exception as e:
    print("[WARN] Porcupine init failed:", e)
    porcupine = None
    audio_stream = None
    pa = None

# Give some time for camera to initialize (not required)
time.sleep(2)

# initialize audio level
audio_level = 0.0

# TTS engine + speak helper
try:
    tts_engine = pyttsx3.init(driverName="sapi5")
except Exception:
    tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
try:
    tts_engine.setProperty("volume", 1.0)
except Exception:
    pass

def speak_response(text: str):
    """Speak text and pause/resume pygame audio to avoid device conflicts."""
    print("Assistant:", text)
    if not text:
        return
    try:
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
        except Exception:
            pass
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("[WARN] TTS failed:", e)
    finally:
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.unpause()
        except Exception:
            pass

assistant_busy = False

def assistant_worker():
    """Background speech-recognize -> match -> speak."""
    global assistant_busy
    if assistant_busy:
        return
    assistant_busy = True
    recognizer = sr.Recognizer()
    try:
        # free mic (stop Porcupine stream) if in use
        try:
            if 'audio_stream' in globals() and audio_stream is not None:
                audio_stream.stop_stream()
        except Exception:
            pass

        with sr.Microphone() as source:
            try:
                print("Listening for user input...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                user_input = recognizer.recognize_google(audio)
                print("User said:", user_input)

                best_cmd, score = best_command_match(user_input, commands)
                if best_cmd:
                    resp = best_cmd.get("response", "")
                    print(f"Matched (score={score:.2f}): {resp}")
                    speak_response(resp)
                else:
                    print(f"No good match (score={score:.2f})")
                    speak_response("No matching response found.")
            except sr.UnknownValueError:
                speak_response("Sorry, I didn't catch that.")
            except sr.RequestError:
                speak_response("Speech recognition failed.")
            except Exception as e:
                print("[WARN] assistant_worker error:", e)
    finally:
        try:
            if 'audio_stream' in globals() and audio_stream is not None:
                audio_stream.start_stream()
        except Exception:
            pass
        assistant_busy = False

while True:
    # Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure eyeAspectRatio has a default value when no faces are detected
    eyeAspectRatio = 0.0

    # Detect facial points through detector function
    faces = detector(gray, 0)

    # Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around each face detected
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect facial points
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        # Use hull to remove convex contour discrepancies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Detect if eye aspect ratio is less than the threshold
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            # If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if ALERT_AVAILABLE:
                    pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                time.sleep(5)
                COUNTER = 0

        else:
            if ALERT_AVAILABLE:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            COUNTER = 0

    # Calculate and display FPS
    new_time = time.time()
    seconds = new_time - _prev_time
    _prev_time = new_time
    fps = 1.0 / seconds if seconds > 0 else 0
    _fps = 0.9 * _fps + 0.1 * fps  # smoothen fps

    # --- Read one audio frame to compute audio level and check wakeword ---
    keyword_index = -1
    try:
        pcm_bytes = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        # compute RMS from raw int16 samples
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size > 0:
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            audio_level = float(np.clip(rms / 32768.0, 0.0, 1.0))
        else:
            audio_level = 0.0
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_bytes)
        keyword_index = porcupine.process(pcm)
    except Exception:
        # on error, keep previous audio_level or set to 0
        audio_level = 0.0
        keyword_index = -1

    # Draw overlay with EAR, status, FPS and audio meter
    if SHOW_OVERLAY:
        frame = draw_overlay(frame, eyeAspectRatio, COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES, _fps, audio_level)

    # Show video feed
    cv2.imshow('Video', frame)

    # handle hotword detection
    if keyword_index >= 0:
        print("Hotword Detected")
        threading.Thread(target=assistant_worker, daemon=True).start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()