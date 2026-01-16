import google.generativeai as genai
import os

# Use the key from gemini_client.py or env
API_KEY = "AIzaSyC6KcojG7D2Uq_lHryo9c3v6wmuDtT9Rm0"
genai.configure(api_key=API_KEY)

print("Available Models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
