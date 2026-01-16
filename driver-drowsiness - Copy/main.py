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

import speech_recognition as sr
import threading
import numpy as np
import os
import math # For animation pulse
from datetime import datetime
import struct
import pvporcupine
import queue
import queue
import pythoncom
import gemini_client
import weather_client
import maps_client
from gui_dashboard import SarathiDashboard

# CONFIG: Set your API Key here or in Environment Variables
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY" 
gemini_client.configure_gemini("AIzaSyAOA45oChaaQFbeuDma7OPfANSdQE9_JQg") # Placeholder, user needs to fill this


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
EYE_ASPECT_RATIO_THRESHOLD = 0.25

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 30

# Counts no. of consecutive frames below threshold value
COUNTER = 0

drowsy_alert_triggered = False
last_alert_time = 0 # Timestamp for non-blocking alert "snooze"
assistant_busy = False

# Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Load the JSON data
with open('cms.json', 'r') as json_file:
    commands = json.load(json_file)["commands"]



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


def draw_overlay(frame, ear, drowsy, fps, audio_level=0.0, wakeword_status=False):
    """Draw semi-transparent status bar, EAR, FPS, instructions and live audio meter on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # --- Animated Alert Background ---
    if drowsy:
        # Calculate pulse alpha (0.0 to 0.6) using sine wave based on time
        pulse = (math.sin(time.time() * 8) + 1) / 2  # oscilates 0 to 1
        alpha_red = 0.6 * pulse
        
        # Red flash overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha_red, frame, 1 - alpha_red, 0, frame)
        
        # Thick flashing border
        if pulse > 0.5:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 30)

        # Force "DROWSY" text to be huge
        cv2.putText(frame, "DROWSY ALERT!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

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

    # Voice Status
    voice_text = "Voice: ON" if wakeword_status else "Voice: OFF"
    voice_col = (0, 255, 0) if wakeword_status else (100, 100, 100)
    cv2.putText(frame, voice_text, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_col, 2)

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
    help_text = "[a] Assistant  [w] Wake  [o] Toggle overlay  [s] Snapshot  [f] Fullscreen  [q] Quit"
    cv2.putText(frame, help_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    return frame

# Initialize Porcupine for wake word detection (robust)
# Porcupine Global State
wakeword_thread_running = True
wakeword_detected = False

def wakeword_listener():
    """Independent thread to listen for wake word without being blocked by video processing."""
    global wakeword_detected, audio_stream, pa, porcupine, wakeword_enabled, assistant_busy

    try:
        keyword_path = os.path.join(os.path.dirname(__file__), 'Hey-Sara_en_windows_v3_0_0.ppn')
        porcupine = pvporcupine.create(
            access_key="nQTUdktIxEO8BWG5sC45JKuafjf7fGtvbOSlEmybAYFPc9wIMhP+6g==",
            keyword_paths=[keyword_path],
            sensitivities=[0.7]
        )
        print(f"[INFO] Porcupine Setup Complete (Threaded).")
        
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        wakeword_enabled = True

    except Exception as e:
        print(f"[ERROR] Porcupine/Audio Init Failed in Thread: {e}")
        wakeword_enabled = False
        return

    print("[INFO] Wake Word Listener Thread Started.")
    
    while wakeword_thread_running:
        if assistant_busy or not wakeword_enabled or audio_stream is None:
            time.sleep(0.1)
            continue
            
        try:
            pcm_bytes = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm_bytes)
            keyword_index = porcupine.process(pcm)
            
            if keyword_index >= 0:
                print("\n[WAKE WORD] Detected via Thread!")
                wakeword_detected = True
                # Small pause to prevent multiple triggers
                time.sleep(1.0) 
        except Exception as e:
            # print(f"[WARN] Audio read error: {e}")
            pass

# Start the audio thread immediately
threading.Thread(target=wakeword_listener, daemon=True).start()

# Give some time for camera to initialize (not required)
time.sleep(1)

# initialize audio level
audio_level = 0.0

# TTS engine + speak helper

# TTS engine + speak helper

def _speak_thread(text, wait_event=None):
    """
    Robust discrete thread for TTS. 
    Initializes a fresh engine instance to avoid COM threading issues.
    """
    try:
        pythoncom.CoInitialize()
        
        # Initialize engine
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        
        # Optional: Adjust mixer if alert is playing
        was_playing = False
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.set_volume(0.2) # Lower volume instead of pause
                was_playing = True
        except:
            pass

        print(f"[DEBUG] Speaking: '{text}'")
        engine.say(text)
        engine.runAndWait()
        
        # Restore volume
        if was_playing:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.set_volume(1.0)
            except:
                pass

    except Exception as e:
        print(f"[ERROR] TTS Thread Failed: {e}")
    finally:
        if wait_event:
            wait_event.set()
        pythoncom.CoUninitialize()

def speak_response(text: str, wait: bool = True):
    """Queue text to be spoken. If wait=True, block until finished."""
    print(f"Assistant: {text}")
    if not text:
        return
    
    # Create a synchronization event if we need to wait
    done_event = threading.Event() if wait else None
    
    # Spawn a fresh thread for this specific utterance
    t = threading.Thread(target=_speak_thread, args=(text, done_event))
    t.start()
    
    # If waiting, block here
    if wait and done_event:
        done_event.wait()





def assistant_worker():
    """Background speech-recognize -> match -> speak."""
    global assistant_busy
    if assistant_busy:
        print("[INFO] Assistant is already busy, ignoring trigger")
        return
    assistant_busy = True
    
    # Speak activation acknowledgment
    speak_response("Yes, I'm listening")
    
    recognizer = sr.Recognizer()
    # Reduce sensitivity
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    
    try:
        # free mic (stop Porcupine stream) if in use
        try:
            if audio_stream is not None:
                audio_stream.stop_stream()
                print("[INFO] Porcupine stream paused for assistant")
        except Exception as e:
            print(f"[WARN] Could not stop audio_stream: {e}")

        with sr.Microphone() as source:
            try:
                print("\n[ASSISTANT] Listening... (speak now)")
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
                print("[ASSISTANT] Processing speech...")
                user_input = recognizer.recognize_google(audio)
                print(f"[ASSISTANT] You said: '{user_input}'")
                
                # --- NEW: Weather & Maps Intents ---
                lower_input = user_input.lower()
                
                # Weather Intent
                if "weather" in lower_input:
                    # simplistic extraction: "weather in [city]"
                    if " in " in lower_input:
                        city = lower_input.split(" in ")[1].strip()
                        resp = weather_client.get_weather(city)
                        speak_response(resp)
                        # Skip CMS/Gemini if handled
                        assistant_busy = False
                        print("[ASSISTANT] Ready\n")
                        return
                    else:
                        speak_response("Which city do you want to know the weather for?")
                        # Could listen again here, but for now just return
                        assistant_busy = False
                        print("[ASSISTANT] Ready\n")
                        return

                # Navigation Intent
                if "navigate to" in lower_input or "take me to" in lower_input:
                    # extract destination
                    if "navigate to" in lower_input:
                        dest = lower_input.split("navigate to")[1].strip()
                    else:
                        dest = lower_input.split("take me to")[1].strip()
                        
                    resp = maps_client.open_navigation(dest)
                    speak_response(resp)
                    assistant_busy = False
                    print("[ASSISTANT] Ready\n")
                    return
                # -----------------------------------

                best_cmd, score = best_command_match(user_input, commands)
                if best_cmd:
                    resp = best_cmd.get("response", "")
                    print(f"[ASSISTANT] Matched (score={score:.2f}): {resp}")
                    speak_response(resp)
                else:
                    print(f"[ASSISTANT] No good match (score={score:.2f}). Trying Gemini...")
                    # Fallback to Gemini
                    llm_response = gemini_client.query_gemini(user_input)
                    if llm_response:
                        print(f"[GEMINI] Response: {llm_response}")
                        speak_response(llm_response)
                    else:
                        speak_response("Sorry, I couldn't find a response and I'm offline.")
            except sr.WaitTimeoutError:
                print("[ASSISTANT] No speech detected (timeout)")
                speak_response("I didn't hear anything.")
            except sr.UnknownValueError:
                print("[ASSISTANT] Could not understand audio")
                speak_response("Sorry, I didn't catch that.")
            except sr.RequestError as e:
                print(f"[ASSISTANT] Speech recognition request failed: {e}")
                speak_response("Speech recognition service failed.")
            except Exception as e:
                print(f"[WARN] assistant_worker error: {e}")
    finally:
        try:
            if audio_stream is not None:
                audio_stream.start_stream()
                print("[INFO] Porcupine stream resumed")
        except Exception as e:
            print(f"[WARN] Could not restart audio_stream: {e}")
        assistant_busy = False
        print("[ASSISTANT] Ready\n")

# --- GLOBAL GUI HANDLE ---
gui = None

def run_core_loop():
    """Entire core loop moved to a thread to keep GUI responsive."""
    global COUNTER, drowsy_alert_triggered, last_alert_time, assistant_busy, gui, _prev_time, _fps, SHOW_OVERLAY, FULLSCREEN, wakeword_enabled, wakeword_detected
    
    print("[INFO] Starting Core Processing Thread...")
    
    while True:
        # Read each frame and flip it, and convert to grayscale
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ensure eyeAspectRatio has a default value when no faces are detected
        eyeAspectRatio = 0.0

        # --- OPTIMIZATION: Resize for faster detection ---
        # Resize to width=320 for detection (2x speedup or more)
        small_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        
        # Detect faces on small frame
        small_faces = detector(small_frame, 0)
        
        # Map detections back to large frame coordinates
        faces = []
        for s_face in small_faces:
            # Scale coordinates x2
            l = s_face.left() * 2
            t = s_face.top() * 2
            r = s_face.right() * 2
            b = s_face.bottom() * 2
            # Create a dlib rectangle for the original frame
            faces.append(dlib.rectangle(l, t, r, b))
        
        # Detect facial points on ORIGINAL frame using scaled rect
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
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    current_time_val = time.time()
                    if not drowsy_alert_triggered or (current_time_val - last_alert_time > 3.0):
                        speak_response("You are drowsy, please wake up!", wait=False)
                        if ALERT_AVAILABLE:
                             if not pygame.mixer.music.get_busy():
                                 pygame.mixer.music.play(-1)
                        drowsy_alert_triggered = True
                        last_alert_time = current_time_val
                    cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                if ALERT_AVAILABLE:
                    try:
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                COUNTER = 0
                drowsy_alert_triggered = False

        # Calculate and display FPS
        new_time = time.time()
        seconds = new_time - _prev_time
        _prev_time = new_time
        fps = 1.0 / seconds if seconds > 0 else 0
        _fps = 0.9 * _fps + 0.1 * fps

        audio_level = 0.0 # Placeholder for now

        # --- Check Thread for Wake Word ---
        if wakeword_detected:
            wakeword_detected = False
            print("[MAIN] Activating Assistant...")
            threading.Thread(target=assistant_worker, daemon=True).start()

        if SHOW_OVERLAY:
            frame = draw_overlay(frame, eyeAspectRatio, COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES, _fps, audio_level, wakeword_enabled)

        # UPDATING THE GUI
        if gui:
            gui.update_video(frame)
            gui.update_stats(assistant_busy, "Listening" if assistant_busy else "Monitoring", is_drowsy=(COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES))

        # Keyboard controls for the Core thread (though GUI should handle this)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quitting via Keyboard...")
            break
        elif key == ord('o'):
            SHOW_OVERLAY = not SHOW_OVERLAY

        
    print("\n[INFO] Cleaning up...")
    video_capture.release()
    cv2.destroyAllWindows()
    if gui:
        gui.quit()

# Entry Point
if __name__ == "__main__":
    # 1. Initialize GUI
    gui = SarathiDashboard()
    
    # 2. Start Core Logic in Thread
    core_thread = threading.Thread(target=run_core_loop, daemon=True)
    core_thread.start()
    
    # 3. Start Assistant Logic (configures gemini)
    gemini_client.configure_gemini()
    
    # 4. Start GUI Main Loop
    print("[INFO] GUI Dashboard Launching...")
    gui.mainloop()

# Clean up Porcupine resources
if audio_stream is not None:
    try:
        audio_stream.stop_stream()
        audio_stream.close()
    except Exception as e:
        print(f"[WARN] Error closing audio stream: {e}")

if porcupine is not None:
    try:
        porcupine.delete()
    except Exception as e:
        print(f"[WARN] Error deleting porcupine: {e}")

if pa is not None:
    try:
        pa.terminate()
    except Exception as e:
        print(f"[WARN] Error terminating PyAudio: {e}")

print("[INFO] Shutdown complete")
