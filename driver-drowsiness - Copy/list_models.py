import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAOA45oChaaQFbeuDma7OPfANSdQE9_JQg")

try:
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
