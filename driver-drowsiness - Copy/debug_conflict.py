import pygame
import pyttsx3
import time
import os

def test_conflict():
    print("Initializing Pygame Mixer...")
    try:
        pygame.mixer.init()
        print("Pygame Mixer Initialized.")
    except Exception as e:
        print(f"Pygame Init Failed: {e}")

    print("Initializing TTS...")
    try:
        engine = pyttsx3.init()
        print("TTS Initialized.")
        
        print("Attempting to speak with mixer ACTIVE...")
        engine.say("This is a test with mixer active.")
        engine.runAndWait()
        print("First phrase spoken.")
        
        time.sleep(1)
        
        print("Quitting mixer...")
        pygame.mixer.quit()
        
        print("Attempting to speak with mixer QUIT...")
        engine.say("This is a test with mixer disabled.")
        engine.runAndWait()
        print("Second phrase spoken.")
        
    except Exception as e:
        print(f"TTS Failed: {e}")

if __name__ == "__main__":
    test_conflict()
